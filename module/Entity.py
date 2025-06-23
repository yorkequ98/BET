class Entity:
    def __init__(self, ID, entity_type, frequency):
        self.id = ID
        assert entity_type in ['user', 'item']
        self.type = entity_type
        self.freq = frequency

    def get_input_form(self):
        return self.freq

    def __str__(self):
        if self.type == 'user':
            return 'u' + str(self.id)
        else:
            return 'i' + str(self.id)

    def __eq__(self, other):
        return self.id == other.id and self.type == other.type