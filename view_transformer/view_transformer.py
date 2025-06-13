class ViewTransformer():
    def __init__(self):
        court_width = 65
        court_length = 105

        self.pixel_vertices = np.array(
            [110,1035],

        )

    def transform(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state