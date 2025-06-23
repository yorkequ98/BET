class Population:
    def __init__(self):
        self.actions = []
        self.max_size = 3

    def add_action(self, action, action_embed, metric):
        action.metric = metric
        self.actions.append((action, action_embed))
        self.actions.sort(key=lambda pair: pair[0].metric, reverse=True)
        # truncate to keep the top-k actions
        self.actions = self.actions[:self.max_size]

    def get_best_action(self, k=1):
        topk_actions = [pair[0] for pair in self.actions[:k]]
        topk_embeddings = [pair[1] for pair in self.actions[:k]]
        return topk_actions, topk_embeddings

    def __len__(self):
        return len(self.actions)
