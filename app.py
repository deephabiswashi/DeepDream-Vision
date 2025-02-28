import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")

# Monkey-patch werkzeug.urls to add url_quote if missing
import werkzeug.urls
if not hasattr(werkzeug.urls, 'url_quote'):
    from urllib.parse import quote as url_quote
    werkzeug.urls.url_quote = url_quote

from flask import Flask, render_template, request, redirect, url_for
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from deepdream import deep_dream, generate_chart, generate_chart_data
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure upload and output folders
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///images.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('database', exist_ok=True)

# Database model for storing user input images
class UserImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize the database within the app context
with app.app_context():
    db.create_all()

# Define the layers to use for generating dream-like outputs.
# (Ensure these layers exist in InceptionV3; 'mixed9' might need tuned hyperparameters.)
DEEPDREAM_LAYERS = ['mixed3', 'mixed5', 'mixed7', 'mixed9']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Store image record in the database
            new_image = UserImage(filename=filename)
            db.session.add(new_image)
            db.session.commit()
            return redirect(url_for('results', filename=filename))
    return render_template('index.html')

@app.route('/results')
def results():
    filename = request.args.get('filename')
    if not filename:
        return redirect(url_for('index'))
    
    input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_paths = {}
    deepdream_results = {}
    
    # Process the image for each selected layer with adjusted hyperparameters for deeper layers.
    for layer in DEEPDREAM_LAYERS:
        output_filename = f"{os.path.splitext(filename)[0]}_{layer}.jpg"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        # For deeper layers like 'mixed9', use more iterations and a smaller step size.
        if layer == 'mixed9':
            dream_img = deep_dream(input_image_path, layer, iterations=30, step_size=0.005, octave_scale=1.2, num_octaves=3)
        else:
            dream_img = deep_dream(input_image_path, layer)
        # Save each output image
        dream_img.save(output_path)
        output_paths[layer] = output_path
        deepdream_results[layer] = dream_img

    # Generate a chart summarizing model activations
    chart_data = generate_chart_data(deepdream_results)
    chart_path = os.path.join('static/images', 'chart.png')
    generate_chart(chart_data, save_path=chart_path)

    # Pass relative URLs to the template
    outputs_relative = {layer: output_paths[layer].split('static/')[1] for layer in output_paths}
    
    return render_template(
        'results.html',
        input_image=url_for('static', filename='uploads/' + filename),
        outputs=outputs_relative, 
        chart=url_for('static', filename='images/chart.png')
    )

if __name__ == '__main__':
    app.run(debug=True)
