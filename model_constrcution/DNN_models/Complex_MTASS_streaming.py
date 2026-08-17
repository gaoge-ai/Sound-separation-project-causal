from DNN_models.Complex_MTASS import Complex_MTASS


class Complex_MTASS_Streaming(Complex_MTASS):
    def __init__(self, is_causal=True):
        super().__init__(is_causal=is_causal)
