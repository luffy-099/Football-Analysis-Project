class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array(
            [110,1035],
            
        )

    def transform(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state