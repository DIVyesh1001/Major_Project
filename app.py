from flask import Flask, request, render_template, redirect, url_for,jsonify,send_file
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

encoder.load_state_dict(torch.load("models/encoder-2-4000.ckpt", map_location='cpu'))
decoder.load_state_dict(torch.load("models/decoder-2-4000.ckpt", map_location='cpu'))

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
        # Check if it's an AJAX request expecting JSON
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
        if "image" not in request.files:
            if is_ajax:
                return jsonify({"error": "No image file provided"}), 400
            return redirect(request.url)
            
        file = request.files["image"]
        if file.filename == "":
            if is_ajax:
                return jsonify({"error": "No image selected"}), 400
            return redirect(request.url)
            
        if file:
            try:
                image = Image.open(file.stream).convert("RGB")
                image = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    features = encoder(image)
                    caption = generate_caption(decoder, features, vocab)
                
                # Clean up the caption by removing special tokens
                caption = clean_caption(caption)
                
                if is_ajax:
                    return jsonify({
                        "caption": caption,
                        "success": True
                    })
                else:
                    # For traditional form submission
                    return render_template("index.html", caption=caption)
            except Exception as e:
                if is_ajax:
                    return jsonify({"error": str(e)}), 500
                return render_template("index.html", error="Error processing image: " + str(e))
    
    # GET request - just render the template
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

def clean_caption(caption):
    """Remove special tokens from the caption"""
    tokens_to_remove = ['<start>', '<end>', '<pad>']
    words = caption.split()
    cleaned_words = [word for word in words if word not in tokens_to_remove]
    return ' '.join(cleaned_words)

# Add a route for example images if needed
# @app.route("/examples/<image_id>")
# def get_example_image(image_id):
#     # Return a predefined example image based on image_id
#     example_images = {
#         "1": "static/examples/1.jpg",
#         "2": "static/examples/2.png",
#         "3": "static/examples/3.png",
#     }
    
#     if image_id in example_images:
#         return send_file(example_images[image_id])
#     else:
#         return "Example image not found", 404

if __name__ == "__main__":
    app.run(debug=True)
