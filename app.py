from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import os

import matplotlib
matplotlib.use('Agg')  # GUIバックエンドを使用しない設定

app = Flask(__name__)

# アップロードされたファイルの保存場所
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# 必要なフォルダを作成
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ホーム画面のルート
@app.route('/')
def home():
    return render_template('home.html')

# 画像アップロードのルート
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('home'))
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return redirect(url_for('home'))

    # Create file path
    filename = image_file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the image file
    image_file.save(filepath)

    # Open image with Pillow
    image = Image.open(filepath)

    # Analyze image and save results
    simplified_image_path, color_distribution_path, brightness_distribution_path, saturation_distribution_path,hue_distribution_path = analyze_image_colors(image, filename)

    # Convert paths to use forward slashes
    original_image_url = filepath.replace(os.path.sep, '/').replace('static/', '')
    simplified_image_url = simplified_image_path.replace(os.path.sep, '/').replace('static/', '')
    color_distribution_url = color_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    brightness_distribution_url = brightness_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    saturation_distribution_url = saturation_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    hue_distribution_url = hue_distribution_path.replace(os.path.sep, '/').replace('static/', '')

    # Render result template with image URLs
    return render_template('result.html', 
                           original_image=original_image_url,  
                           simplified_image_path=simplified_image_url,  
                           color_distribution_path=color_distribution_url,
                           brightness_distribution_path=brightness_distribution_url,
                           saturation_distribution_path=saturation_distribution_url,
                           hue_distribution_path=hue_distribution_url)

# 画像の色分析を行う関数
def analyze_image_colors(image, filename, max_size=500):
    # Convert image to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image while maintaining aspect ratio
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_height = max_size
        new_width = int((max_size / height) * width)
    
    image = image.resize((new_width, new_height))  # Resize with aspect ratio
    img_data = np.array(image)

    # Convert RGB to HSV and block color values
    hsv_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV)
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    h = (h // 3) * 3  # Group hues into blocks
    s = (s // 8) * 8  # Group saturation into blocks
    v = (v // 8) * 8  # Group value (brightness) into blocks
    
    # Create simplified RGB image
    simplified_hsv_image = np.stack([h, s, v], axis=-1)
    simplified_rgb_image = cv2.cvtColor(simplified_hsv_image, cv2.COLOR_HSV2RGB)

    # Save simplified image
    simplified_image_path = os.path.join(app.config['RESULT_FOLDER'], 'simplified_' + filename)
    Image.fromarray(simplified_rgb_image).save(simplified_image_path)

    # Call functions to create color, brightness, saturation, and hue distributions
    color_distribution_path = create_color_distribution(simplified_rgb_image, filename)
    brightness_distribution_path = create_brightness_distribution(img_data, filename)
    saturation_distribution_path = create_saturation_distribution(hsv_image, filename)
    hue_distribution_path = create_hue_distribution(hsv_image, filename)

    return simplified_image_path, color_distribution_path, brightness_distribution_path, saturation_distribution_path, hue_distribution_path


# 色分布を生成して保存する関数
def create_color_distribution(image, filename):
    reshaped_img = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5)  # 5つのクラスターを作成
    kmeans.fit(reshaped_img)

    # クラスタ中心（頻出色）と割合を取得
    top_colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    # 色の割合を計算
    percentages = counts / counts.sum()

    # 色分布を棒グラフで表示して保存
    color_distribution_path = os.path.join(app.config['RESULT_FOLDER'], 'color_distribution_' + filename + '.png')
    plt.figure(figsize=(6, 4))
    plt.bar(range(5), percentages, color=[top_colors[i] / 255 for i in range(5)])
    plt.title("Color Distribution")
    plt.xticks(range(5), [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in top_colors], rotation=45)
    plt.tight_layout()
    plt.savefig(color_distribution_path)
    plt.close()

    return color_distribution_path

# 明度（輝度）分布を生成して保存する関数
def create_brightness_distribution(image, filename):
    brightness = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([brightness], [0], None, [256], [0, 256])

    # ヒストグラムを曲線グラフで表示して保存
    brightness_curve_path = os.path.join(app.config['RESULT_FOLDER'], 'brightness_curve_' + filename + '.png')
    plt.figure(figsize=(8, 4))
    plt.plot(hist, color='black')
    plt.title("Brightness Distribution")
    plt.xlabel("Brightness Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(brightness_curve_path)
    plt.close()

    return brightness_curve_path

# 彩度（サチュレーション）分布を生成して保存する関数
def create_saturation_distribution(hsv_image, filename):
    s = hsv_image[:, :, 1]
    hist = cv2.calcHist([s], [0], None, [256], [0, 256])

    # ヒストグラムを曲線グラフで表示して保存
    saturation_distribution_path = os.path.join(app.config['RESULT_FOLDER'], 'saturation_distribution_' + filename + '.png')
    plt.figure(figsize=(8, 4))
    plt.plot(hist, color='blue')
    plt.title("Saturation Distribution")
    plt.xlabel("Saturation Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(saturation_distribution_path)
    plt.close()

    return saturation_distribution_path

def create_hue_distribution(hsv_image, filename):
    h = hsv_image[:, :, 0]  # 色相チャネルを抽出
    hue_hist, bin_edges = np.histogram(h, bins=180, range=(0, 180))  # 色相のヒストグラムを作成

    # 色相を12時方向から開始するように調整 (黄色が一番上)
    bin_edges_shifted = (bin_edges) % 180  # 色相を-30度シフトして黄色を上に
    
    # 極座標プロット用の角度を計算
    theta = np.linspace(0, 2 * np.pi, 180)
    
    # 極座標プロット
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    
    # 棒グラフとして色相の分布をプロット
    ax.bar(theta, hue_hist, width=2 * np.pi / 180, color=plt.cm.hsv(bin_edges_shifted[:-1] / 180.0), bottom=0.0)
    
    # ラベルや目盛りを削除し、色相環を視覚的に際立たせる
    # ax.set_xticks([])
    ax.set_yticks([])
    # ax.spines['polar'].set_visible(False)  # 外枠を削除
    
    # タイトルと保存パス
    ax.set_title('Hue Distribution Wheel', va='bottom')
    hue_distribution_path = os.path.join(app.config['RESULT_FOLDER'], 'hue_wheel_' + filename + '.png')
    plt.savefig(hue_distribution_path)
    plt.close()
    
    return hue_distribution_path



if __name__ == '__main__':
    app.run(debug=True)
