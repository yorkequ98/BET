import torch
import torch.nn as nn
from module import config
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=16):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        return x


class SetAggregator(nn.Module):
    """
    Aggregating the set embeddings to form the embedding of an action
    """
    def __init__(self, dim):
        super(SetAggregator, self).__init__()
        self.dim = dim

    def forward(self, x):
        output = torch.mean(x, dim=self.dim)
        return output


class EntityAggregator(nn.Module):
    """
    Aggregating the element embeddings to form the embedding of a set
    """
    def __init__(self, dim):
        super(EntityAggregator, self).__init__()
        self.dim = dim

    def forward(self, x, mask, num_nonzeros):
        # mask: (128, size of the largest set, 16)
        # Apply binary mask to input tensor
        masked_x = x * mask

        # Sum the masked input tensor along the element dimension
        aggregate_x = torch.mean(masked_x, dim=self.dim)

        # Compute the number of non-zero elements in the mask
        num_sets, element_emb_size = aggregate_x.size()
        assert len(num_nonzeros) == num_sets
        divident = torch.repeat_interleave(num_nonzeros, element_emb_size).reshape((num_sets, element_emb_size))

        # Average the sum over the number of non-zero elements
        output = aggregate_x / divident
        return output


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.leaky_relu = nn.LeakyReLU()
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        return x


class DeepSets(nn.Module):
    def __init__(self, user_freqs, item_freqs):
        super(DeepSets, self).__init__()
        input_size = 1
        output_size = 1
        element_emb_size = 16
        action_emb_size = 64

        # encoder1 is used to encode each user
        self.encoder1 = Encoder(input_size, element_emb_size)
        # encoder2 is used to encode each item
        self.encoder2 = Encoder(input_size, element_emb_size)
        # encoder3 is used to encode each set
        self.encoder3 = Encoder(element_emb_size + 1, action_emb_size, hidden_size=action_emb_size)
        # aggregator1 is used to fuse users/items to obtain set embedding
        self.aggregator1 = EntityAggregator(1)
        # aggregator2 is used to fuse each set embedding to obtain action embedding
        self.aggregator2 = SetAggregator(0)
        # decoder is used to evaluate the action
        self.decoder = Decoder(action_emb_size, output_size)

        self.user_freqs = torch.tensor(user_freqs, device=config.device)
        self.user_freqs = torch.unsqueeze(self.user_freqs, dim=1).float()
        self.item_freqs = torch.tensor(item_freqs, device=config.device)
        self.item_freqs = torch.unsqueeze(self.item_freqs, dim=1).float()

        self.criterion = nn.MSELoss()
        self.critic_opt = optim.Adam(params=self.parameters(), lr=0.01)

    def get_action_embeddings(self, users, items, mask, num_nonzeros):
        """
        Format of the input x: 128 * 2 * max_set_size
        [
            [u1, u2],
            [u3, u4],
            [u5, ∅]]
        ]
        """
        # encode each user and item to derive their embeddings
        user_emb = self.encoder1(users.unsqueeze(-1).float())
        item_emb = self.encoder2(items.unsqueeze(-1).float())

        # concatenate user and item embeddings
        selected_element_emb = torch.cat((user_emb, item_emb), dim=1)
        set_emb = self.aggregator1(selected_element_emb, mask, num_nonzeros)

        assert len(set_emb) == len(range(config.MIN_EMB_SIZE, config.MAX_EMB_SIZE + 1))

        # normalise sizes to [0, 1]
        diff = torch.arange(config.MIN_EMB_SIZE, config.MAX_EMB_SIZE + 1) - config.MIN_EMB_SIZE
        sizes = diff / (config.MAX_EMB_SIZE - config.MIN_EMB_SIZE)
        sizes = sizes.to(config.device).unsqueeze(-1).float()

        # concat size and set embeddings
        extended_set_emb = torch.cat((set_emb, sizes), dim=1).float()

        # encode sets and fuse set embeddings to derive action embeddings
        enc_extended_set_emb = self.encoder3(extended_set_emb)

        action_embeddings = self.aggregator2(enc_extended_set_emb)
        return action_embeddings

    def forward(self, users, items, mask, num_nonzeros):
        action_embeddings = self.get_action_embeddings(users, items, mask, num_nonzeros)
        scores = self.decoder(action_embeddings)
        return action_embeddings, scores

    def update_critic(self, buffer):
        n = 2
        losses = []
        for _ in range(n):
            # calculate critic loss and update the critic
            buffered_action, buffered_metric = buffer.sample()
            users, items, mask, num_nonzeros = buffered_action.get_set_form()
            _, predicted_score = self.forward(users, items, mask, num_nonzeros)
            critic_loss = self.criterion(buffered_metric, predicted_score)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            # update learning rate
            self.critic_opt.param_groups[0]['lr'] = max(0.002, 0.9 * self.critic_opt.param_groups[0]['lr'])

            txt = 'Buffered RQ: {:.4f}, predicted RQ: {:.4f}, loss: {:.4f}'
            print(txt.format(buffered_metric.item(), predicted_score.item(), critic_loss.item()))
            losses.append(critic_loss.item())
        return losses




