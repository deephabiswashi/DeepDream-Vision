# <p align="center">DeepDream-Vision</p>

<p align="center"><em>CVPR-inspired DeepDream Visualizations using InceptionV3</em></p>

<hr style="border: 1px solid #ccc; margin: 20px 0;" />

## 1. Overview

DeepDream-Vision is a project that leverages a pre-trained InceptionV3 model to generate artistic, dream-like images from user-supplied inputs. By applying the DeepDream technique on different convolutional layers (e.g., `mixed3`, `mixed5`, `mixed7`, and `mixed9`), the project produces multiple outputs that highlight both low-level textures and high-level features. It also includes:
- A Flask-based backend with a SQLite database to store uploaded images.
- An interactive, 3D-styled UI with a theme toggle.
- Visualization charts of model activations.

<hr style="border: 1px solid #ccc; margin: 20px 0;" />

## 2. Project Structure

```
DeepDream-Vision/
├── README.md               # Project description
├── requirements.txt        # Python dependencies
├── app.py                  # Flask web application
├── deepdream.py            # DeepDream model implementation
├── database/
│   └── images.db           # SQLite database for uploaded images
├── static/
│   ├── css/
│   │   └── style.css       # UI styles (3D effects, theme toggle, etc.)
│   ├── js/
│   │   └── script.js       # JavaScript for UI interactivity and theme toggling
│   ├── uploads/            # User-uploaded images
│   ├── outputs/            # Generated DeepDream images
│   └── images/             # Generated charts and additional graphics
└── templates/
    ├── index.html          # Image upload page
    └── results.html        # Results page displaying outputs and charts
```

<hr style="border: 1px solid #ccc; margin: 20px 0;" />

## 3. Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/deephabiswashi/DeepDream-Vision.git
   cd DeepDream-Vision
   ```

2. **Create and Activate a Virtual Environment (optional):**

   ```bash
   python -m venv env
   source env/bin/activate    # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

<hr style="border: 1px solid #ccc; margin: 20px 0;" />

## 4. Usage

1. **Run the Application:**

   ```bash
   python3 app.py
   ```

2. **Access the Web Interface:**
   
   Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to upload an image and generate dream-like outputs.

3. **Features:**
   - **Upload Page:** Submit an image using the interactive 3D-styled UI with a vibrant/dark theme toggle.
   - **Results Page:** View the original image alongside multiple DeepDream outputs (generated from different layers), download or copy the output URLs, and see an activation chart.

<hr style="border: 1px solid #ccc; margin: 20px 0;" />

## 5. Additional Details

- **Output Resolution:**  
  The model preserves the original image resolution so that outputs match the uploaded image's size.

- **Hyperparameters:**  
  Hyperparameters (e.g., iterations, step sizes) are adjustable in `deepdream.py` for each layer, ensuring optimal outputs across different layers.

- **Interactive UI:**  
  The UI includes subtle 3D effects, smooth transitions, and a theme toggle button for a modern, engaging user experience.

<hr style="border: 1px solid #ccc; margin: 20px 0;" />

## 6. Screenshots

### Homepage
![Homepage](screenshots/homepage.png)

### Results Page
![Results Page](screenshots/results-page.png)

### Output Images
#### Output Image 1
![Output Image 1](screenshots/output_img1.png)

#### Output Image 2
![Output Image 3](screenshots/output_img3.png)

#### Output Image 3
![Output Image 5](screenshots/output_img5.png)


## 7. Contributing

Contributions are welcome! Please open an issue or submit a pull request to improve the project or fix bugs.

<hr style="border: 1px solid #ccc; margin: 20px 0;" />

## 8. License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

<hr style="border: 1px solid #ccc; margin: 20px 0;" />

<p align="center"><em>Made with ❤️ by Deep Habiswashi</em></p>