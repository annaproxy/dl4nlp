'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
class GPT2Config(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=20000,
            n_positions=30,
            n_ctx=30,
            n_embd=256,
            n_layer=2,
            n_head=2,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
