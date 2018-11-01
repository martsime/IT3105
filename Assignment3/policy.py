import math


class Policy:
    class Tree:
        @staticmethod
        def get(name):
            if name == 'utc_wiki':
                return Policy.Tree.utc_wiki
            elif name == 'utc_lecture':
                return Policy.Tree.utc_lecture
            else:
                raise ValueError(f'Invalid tree policy: "{name}"')

        @staticmethod
        def utc_wiki(node):
            """UTC Algorithm from wikipedia"""
            c = 1
            if not node.has_children:
                return None

            if node.traversals == 0:
                return tuple(node.children.keys())[0]
            
            best_action = None
            best_score = - math.inf

            for action in node.children.keys():
                child = node.children[action]
                if child.traversals == 0:
                    return action

                child_probability = child.get_probability()

                # https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
                child_score = child_probability + c * math.sqrt(math.log(node.traversals, math.e) / child.traversals)
                
                if child_score > best_score:
                    best_score = child_score
                    best_action = action

            return best_action
        
        @staticmethod
        def utc_lecture(node):
            """UTC Algorithm from lecture slides"""
            c = 1
            if not node.has_children:
                return None

            if node.traversals == 0:
                return tuple(node.children.keys())[0]

            best_action = None
            best_score = - math.inf
            for action in node.children.keys():
                child = node.children[action]
                child_score = c * math.sqrt(math.log(node.traversals, math.e) / (child.traversals + 1))

                if child_score > best_score:
                    best_score = child_score
                    best_action = action

            return best_action

        @staticmethod
        def best(node):
            """"Returns the best action (explotation)"""
            if not node.has_children:
                return None

            best_action = None
            best_prob = - math.inf
            for action in node.children.keys():
                child = node.children[action]
                prob = child.get_probability() * 100
                if prob > best_prob:
                    best_action = action
                    best_prob = prob

            return best_action

        @staticmethod
        def worst(node):
            """"Returns the worst action (explotation)"""
            if not node.has_children:
                return None

            keys = tuple(node.children.keys())
            worst_action = keys[0]
            worst_prob = node.children.get(worst_action).get_probability() * 100
            for action in keys[1:]:
                child = node.children[action]
                prob = child.get_probability() * 100
                if prob < worst_prob:
                    worst_action = action
                    worst_prob = prob

            return worst_action

    class Score:
        @staticmethod
        def get(name):
            if name == 'zero_one':
                return Policy.Score.one_zero
            else:
                raise ValueError(f'Invalid score policy: "{name}"')

        @staticmethod
        def one_zero(win=True):
            if win:
                return 1.0
            else:
                return 0.0

