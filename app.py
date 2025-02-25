from flask import Flask, request, render_template, redirect, url_for
import torch
from model import EncoderCNN, DecoderRNN
import pickle
from build_vocab import Vocabulary
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

embed_size = 256
hidden_size = 512
num_layers = 1

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)

encoder.load_state_dict(torch.load("models/encoder-1-18000.ckpt", map_location='cpu'))
decoder.load_state_dict(torch.load("models/decoder-1-18000.ckpt", map_location='cpu'))

encoder.eval()
decoder.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            image = Image.open(file.stream).convert("RGB")
            image = transform(image).unsqueeze(0)
            
            
            with torch.no_grad():
                features = encoder(image)
                caption = generate_caption(decoder, features, vocab)
            
            return render_template("result.html", caption=caption)
    return render_template("index.html")

def generate_caption(decoder, features, vocab, max_length=20):
    sampled_ids = decoder.sample(features) 
    sampled_ids = sampled_ids[0].cpu().numpy()
    caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        caption.append(word)
        if word == '<end>':
            break
    return ' '.join(caption)

if __name__ == "__main__":
    app.run(debug=True)
