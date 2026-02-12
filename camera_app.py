"""
YOLO物体検出カメラアプリ
gui_app.pyで学習したモデルを使用してリアルタイム物体検出を行う
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
from pathlib import Path
import yaml

# ultralyticsのインポート
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Detection will be disabled.")


class CameraDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 物体検出カメラ")
        self.root.geometry("1000x700")

        # 基本パス設定
        self.base_path = Path(__file__).parent.resolve()
        self.model_path = self.base_path / "model"
        self.data_yaml_path = self.base_path / "data.yaml"

        # モデル関連
        self.model = None
        self.classes = self.load_classes()

        # カメラ関連
        self.cap = None
        self.camera_running = False

        # 検出設定
        self.confidence_threshold = 0.5
        self.detection_enabled = True

        # UIセットアップ
        self.setup_ui()

    def load_classes(self):
        """data.yamlからクラス情報を読み込む"""
        if self.data_yaml_path.exists():
            with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    if isinstance(data['names'], dict):
                        return list(data['names'].values())
                    return data['names']
        return ["object"]

    def setup_ui(self):
        """UIのセットアップ"""
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左側：カメラプレビュー
        left_frame = ttk.LabelFrame(main_frame, text="カメラ映像")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # カメラキャンバス
        self.camera_canvas = tk.Canvas(left_frame, width=800, height=600, bg='black')
        self.camera_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 右側：コントロール
        right_frame = ttk.Frame(main_frame, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        right_frame.pack_propagate(False)

        # モデル選択
        model_frame = ttk.LabelFrame(right_frame, text="モデル設定")
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_frame, text="モデルファイル:").pack(anchor=tk.W, padx=5, pady=2)

        model_select_frame = ttk.Frame(model_frame)
        model_select_frame.pack(fill=tk.X, padx=5, pady=2)

        self.model_path_var = tk.StringVar()
        self.model_entry = ttk.Entry(model_select_frame, textvariable=self.model_path_var)
        self.model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(model_select_frame, text="...", width=3, command=self.browse_model).pack(side=tk.RIGHT)

        ttk.Button(model_frame, text="モデル読み込み", command=self.load_model).pack(fill=tk.X, padx=5, pady=5)

        # モデル状態表示
        self.model_status_var = tk.StringVar(value="モデル未読み込み")
        ttk.Label(model_frame, textvariable=self.model_status_var, foreground="red").pack(padx=5, pady=2)

        # カメラ設定
        camera_frame = ttk.LabelFrame(right_frame, text="カメラ設定")
        camera_frame.pack(fill=tk.X, pady=5)

        camera_id_frame = ttk.Frame(camera_frame)
        camera_id_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(camera_id_frame, text="カメラID:").pack(side=tk.LEFT)
        self.camera_id_var = tk.StringVar(value="0")
        ttk.Entry(camera_id_frame, textvariable=self.camera_id_var, width=5).pack(side=tk.LEFT, padx=5)

        camera_btn_frame = ttk.Frame(camera_frame)
        camera_btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.camera_start_btn = ttk.Button(camera_btn_frame, text="カメラ開始", command=self.start_camera)
        self.camera_start_btn.pack(side=tk.LEFT, padx=2)

        self.camera_stop_btn = ttk.Button(camera_btn_frame, text="カメラ停止", command=self.stop_camera, state=tk.DISABLED)
        self.camera_stop_btn.pack(side=tk.LEFT, padx=2)

        # 検出設定
        detect_frame = ttk.LabelFrame(right_frame, text="検出設定")
        detect_frame.pack(fill=tk.X, pady=5)

        # 信頼度閾値
        ttk.Label(detect_frame, text="信頼度閾値:").pack(anchor=tk.W, padx=5, pady=2)

        conf_frame = ttk.Frame(detect_frame)
        conf_frame.pack(fill=tk.X, padx=5, pady=2)

        self.conf_var = tk.DoubleVar(value=0.5)
        self.conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, variable=self.conf_var,
                                    orient=tk.HORIZONTAL, command=self.on_conf_change)
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.conf_label = ttk.Label(conf_frame, text="0.50", width=5)
        self.conf_label.pack(side=tk.RIGHT)

        # 検出ON/OFF
        self.detect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(detect_frame, text="検出を有効化", variable=self.detect_var).pack(anchor=tk.W, padx=5, pady=5)

        # 検出結果
        result_frame = ttk.LabelFrame(right_frame, text="検出結果")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.result_text = tk.Text(result_frame, height=15, width=30)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)

        # クラス情報
        class_frame = ttk.LabelFrame(right_frame, text="クラス情報")
        class_frame.pack(fill=tk.X, pady=5)

        self.class_info_label = ttk.Label(class_frame, text=f"クラス数: {len(self.classes)}")
        self.class_info_label.pack(padx=5, pady=2)

        classes_text = ", ".join(self.classes[:5])
        if len(self.classes) > 5:
            classes_text += f" ... (+{len(self.classes) - 5})"
        ttk.Label(class_frame, text=classes_text, wraplength=200).pack(padx=5, pady=2)

        # 自動でモデルを探す
        self.auto_find_model()

    def auto_find_model(self):
        """モデルディレクトリから最新のモデルを自動検索"""
        if not self.model_path.exists():
            return

        # model/exp*/weights/best.pt を探す
        best_models = list(self.model_path.glob("**/weights/best.pt"))
        if best_models:
            # 最新のものを選択（更新日時でソート）
            latest_model = max(best_models, key=lambda p: p.stat().st_mtime)
            self.model_path_var.set(str(latest_model))

    def browse_model(self):
        """モデルファイルを選択"""
        filepath = filedialog.askopenfilename(
            title="モデルファイルを選択",
            initialdir=str(self.model_path),
            filetypes=[("PyTorch Model", "*.pt")]
        )
        if filepath:
            self.model_path_var.set(filepath)

    def load_model(self):
        """モデルを読み込む"""
        if not ULTRALYTICS_AVAILABLE:
            messagebox.showerror("エラー", "ultralyticsがインストールされていません")
            return

        model_file = self.model_path_var.get()
        if not model_file or not os.path.exists(model_file):
            messagebox.showerror("エラー", "有効なモデルファイルを選択してください")
            return

        try:
            self.model = YOLO(model_file)
            self.model_status_var.set("モデル読み込み完了")
            messagebox.showinfo("完了", f"モデルを読み込みました:\n{model_file}")

            # モデルのクラス情報を取得
            if hasattr(self.model, 'names') and self.model.names:
                self.classes = list(self.model.names.values())
                self.class_info_label.config(text=f"クラス数: {len(self.classes)}")

        except Exception as e:
            messagebox.showerror("エラー", f"モデル読み込みエラー:\n{e}")
            self.model = None
            self.model_status_var.set("モデル読み込み失敗")

    def on_conf_change(self, value):
        """信頼度閾値変更時"""
        self.confidence_threshold = float(value)
        self.conf_label.config(text=f"{self.confidence_threshold:.2f}")

    def start_camera(self):
        """カメラを開始"""
        try:
            camera_id = int(self.camera_id_var.get())
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                messagebox.showerror("エラー", f"カメラID {camera_id} を開けませんでした")
                return

            self.camera_running = True
            self.camera_start_btn.config(state=tk.DISABLED)
            self.camera_stop_btn.config(state=tk.NORMAL)
            self.update_camera()
        except Exception as e:
            messagebox.showerror("エラー", f"カメラ開始エラー: {e}")

    def stop_camera(self):
        """カメラを停止"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_start_btn.config(state=tk.NORMAL)
        self.camera_stop_btn.config(state=tk.DISABLED)
        self.camera_canvas.delete("all")

    def update_camera(self):
        """カメラフレームを更新"""
        if self.camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # 物体検出を実行
                if self.model and self.detect_var.get():
                    frame, detections = self.detect_objects(frame)
                    self.update_results(detections)

                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # キャンバスサイズに合わせてリサイズ
                canvas_w = self.camera_canvas.winfo_width()
                canvas_h = self.camera_canvas.winfo_height()

                if canvas_w > 1 and canvas_h > 1:
                    h, w = frame_rgb.shape[:2]
                    scale = min(canvas_w / w, canvas_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

                    # PhotoImageに変換
                    img = Image.fromarray(frame_resized)
                    self.camera_photo = ImageTk.PhotoImage(img)

                    # キャンバスに描画
                    self.camera_canvas.delete("all")
                    self.camera_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.camera_photo)

            self.root.after(30, self.update_camera)

    def detect_objects(self, frame):
        """物体検出を実行"""
        detections = []

        try:
            # 推論実行
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)

            # 結果を描画
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # バウンディングボックス座標
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])

                        # クラス名取得
                        cls_name = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"

                        # 検出結果を記録
                        detections.append({
                            'class': cls_name,
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2)
                        })

                        # バウンディングボックスを描画
                        color = self.get_color(cls_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # ラベルを描画
                        label = f"{cls_name}: {conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except Exception as e:
            print(f"Detection error: {e}")

        return frame, detections

    def get_color(self, class_id):
        """クラスIDに基づいて色を取得"""
        colors = [
            (255, 0, 0),    # 赤
            (0, 255, 0),    # 緑
            (0, 0, 255),    # 青
            (255, 255, 0),  # シアン
            (255, 0, 255),  # マゼンタ
            (0, 255, 255),  # 黄
            (128, 0, 255),  # 紫
            (255, 128, 0),  # オレンジ
        ]
        return colors[class_id % len(colors)]

    def update_results(self, detections):
        """検出結果を更新"""
        self.result_text.delete(1.0, tk.END)

        if not detections:
            self.result_text.insert(tk.END, "検出なし\n")
            return

        self.result_text.insert(tk.END, f"検出数: {len(detections)}\n")
        self.result_text.insert(tk.END, "-" * 30 + "\n")

        for i, det in enumerate(detections):
            self.result_text.insert(tk.END, f"{i + 1}. {det['class']}\n")
            self.result_text.insert(tk.END, f"   信頼度: {det['confidence']:.2%}\n")
            x1, y1, x2, y2 = det['bbox']
            self.result_text.insert(tk.END, f"   位置: ({x1}, {y1}) - ({x2}, {y2})\n")

    def on_closing(self):
        """アプリ終了時の処理"""
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CameraDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
