class Settings:
    @staticmethod
    def nim():
        return {
            'game': 'nim',
            'N': 99,
            'K': 6,
            'P': 1,
            'G': 3,
            'M': 1000,
            'verbose': True,
            'tree_policy': 'utc_wiki',
            'score_policy': 'zero_one',
        }
    @staticmethod
    def tictactoe():
        return {
            'game': 'tictactoe',
            'P': 'random',
            'G': 10,
            'M': 1000,
            'verbose': True,
            'tree_policy': 'utc_wiki',
            'score_policy': 'zero_one',
        }
