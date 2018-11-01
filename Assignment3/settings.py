class Settings:
    @staticmethod
    def hex():
        return {
            'game': 'hex',
            'size': 5,
            'P': 1,
            'G': 1,
            'M': 1000,
            'verbose': True,
            'tree_policy': 'utc_wiki',
            'score_policy': 'zero_one',
        }
