from node import Node

class NodeManager:
    def __init__(self):
        self.nodes = {}

    def get_node(self, state):
        key = repr(state)
        if key in self.nodes:
            return self.nodes.get(key)
        else:
            new_node = Node(state=state)
            self.nodes[key] = new_node
            return new_node

    def expand_node(self, node):
        for action in node.state.get_actions():
            new_state = node.state.get_copy()
            new_state.do_action(action)
            key = repr(new_state)
            if key in self.nodes:
                new_node = self.nodes.get(key)
            else:
                new_node = Node(state=new_state)
                self.nodes[key] = new_node
            
            node.children[action] = new_node





    

