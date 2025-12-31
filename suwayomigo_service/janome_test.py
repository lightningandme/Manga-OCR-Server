from janome.tokenizer import Tokenizer

t = Tokenizer()
text = "学校に行かなかった"

for token in t.tokenize(text):
    print(f"表面形: {token.surface} | 原型: {token.base_form} | 词性: {token.part_of_speech}")