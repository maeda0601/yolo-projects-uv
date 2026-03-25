"""
YOLO画像収集・学習GUIアプリ
tkinterを使用して、画像収集、アノテーション、YOLO学習をGUIで操作できるアプリ
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import shutil
import threading
import yaml
from datetime import datetime
from pathlib import Path

# ultralyticsのインポート
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Training will be disabled.")


class YOLOGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 画像収集・学習 GUI")
        self.root.geometry("1200x800")

        # 基本パス設定（スクリプトのディレクトリを基準にする）
        self.base_path = Path(__file__).parent.resolve()
        self.dataset_path = self.base_path / "dataset"
        self.data_yaml_path = self.base_path / "data.yaml"
        self.model_path = self.base_path / "model"

        # ディレクトリ作成
        (self.dataset_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.dataset_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.dataset_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.dataset_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # クラス情報
        self.classes = self.load_classes()

        # カメラ関連
        self.cap = None
        self.camera_running = False

        # アノテーション関連
        self.current_image_path = None
        self.current_image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.bboxes = []  # [(class_id, x_center, y_center, width, height), ...]
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None
        self.selected_class_id = 0

        # 学習関連
        self.training = False
        self.training_thread = None

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
        return ["object"]  # デフォルトクラス

    def save_classes(self):
        """data.yamlにクラス情報を保存"""
        data = {
            'path': './dataset',  # 相対パスを使用（汎用性のため）
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(self.classes)}
        }
        with open(self.data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    def setup_ui(self):
        """UIのセットアップ"""
        # メインノートブック（タブ）
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 各タブを作成
        self.create_collection_tab()
        self.create_annotation_tab()
        self.create_training_tab()
        self.create_class_management_tab()

    # ==================== 画像収集タブ ====================
    def create_collection_tab(self):
        """画像収集タブを作成"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="画像収集")

        # 左側：カメラプレビュー
        left_frame = ttk.LabelFrame(tab, text="カメラ撮影")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # カメラプレビューキャンバス
        self.camera_canvas = tk.Canvas(left_frame, width=640, height=480, bg='black')
        self.camera_canvas.pack(padx=5, pady=5)

        # カメラコントロール
        camera_ctrl = ttk.Frame(left_frame)
        camera_ctrl.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(camera_ctrl, text="カメラID:").pack(side=tk.LEFT)
        self.camera_id_var = tk.StringVar(value="0")
        ttk.Entry(camera_ctrl, textvariable=self.camera_id_var, width=5).pack(side=tk.LEFT, padx=5)

        self.camera_start_btn = ttk.Button(camera_ctrl, text="カメラ開始", command=self.start_camera)
        self.camera_start_btn.pack(side=tk.LEFT, padx=5)

        self.camera_stop_btn = ttk.Button(camera_ctrl, text="カメラ停止", command=self.stop_camera, state=tk.DISABLED)
        self.camera_stop_btn.pack(side=tk.LEFT, padx=5)

        # 保存先選択
        save_frame = ttk.Frame(left_frame)
        save_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(save_frame, text="保存先:").pack(side=tk.LEFT)
        self.save_dest_var = tk.StringVar(value="train")
        ttk.Radiobutton(save_frame, text="Train", variable=self.save_dest_var, value="train").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(save_frame, text="Val", variable=self.save_dest_var, value="val").pack(side=tk.LEFT, padx=10)

        # プレフィックス
        prefix_frame = ttk.Frame(left_frame)
        prefix_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(prefix_frame, text="ファイル名プレフィックス:").pack(side=tk.LEFT)
        self.prefix_var = tk.StringVar(value="img")
        ttk.Entry(prefix_frame, textvariable=self.prefix_var, width=20).pack(side=tk.LEFT, padx=5)

        # 撮影ボタン
        self.capture_btn = ttk.Button(left_frame, text="撮影", command=self.capture_image, state=tk.DISABLED)
        self.capture_btn.pack(pady=10)

        # 右側：ファイル読み込み
        right_frame = ttk.LabelFrame(tab, text="ファイル読み込み")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Button(right_frame, text="画像ファイルを選択", command=self.import_images).pack(pady=10)

        # インポートログ
        self.import_log = tk.Text(right_frame, height=20, width=40)
        self.import_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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
            self.capture_btn.config(state=tk.NORMAL)
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
        self.capture_btn.config(state=tk.DISABLED)
        self.camera_canvas.delete("all")

    def update_camera(self):
        """カメラフレームを更新"""
        if self.camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # リサイズ
                h, w = frame_rgb.shape[:2]
                canvas_w = self.camera_canvas.winfo_width()
                canvas_h = self.camera_canvas.winfo_height()
                scale = min(canvas_w/w, canvas_h/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

                # PhotoImageに変換
                img = Image.fromarray(frame_resized)
                self.camera_photo = ImageTk.PhotoImage(img)

                # キャンバスに描画
                self.camera_canvas.delete("all")
                self.camera_canvas.create_image(canvas_w//2, canvas_h//2, image=self.camera_photo)

            self.root.after(30, self.update_camera)

    def capture_image(self):
        """画像を撮影して保存"""
        if hasattr(self, 'current_frame'):
            dest = self.save_dest_var.get()
            prefix = self.prefix_var.get()

            # 既存ファイル数をカウントして連番を決定
            save_dir = self.dataset_path / "images" / dest
            existing_files = list(save_dir.glob(f"{prefix}_*.jpg"))
            next_num = len(existing_files) + 1

            # ファイル名生成
            filename = f"{prefix}_{next_num:04d}.jpg"
            filepath = save_dir / filename

            # 保存
            cv2.imwrite(str(filepath), self.current_frame)
            self.import_log.insert(tk.END, f"保存: {filepath}\n")
            self.import_log.see(tk.END)

    def import_images(self):
        """画像ファイルをインポート"""
        files = filedialog.askopenfilenames(
            title="画像ファイルを選択",
            filetypes=[("画像ファイル", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not files:
            return

        dest = self.save_dest_var.get()
        prefix = self.prefix_var.get()
        save_dir = self.dataset_path / "images" / dest

        # 既存ファイル数をカウント
        existing_files = list(save_dir.glob(f"{prefix}_*.jpg"))
        next_num = len(existing_files) + 1

        for file in files:
            filename = f"{prefix}_{next_num:04d}.jpg"
            filepath = save_dir / filename

            # 画像を読み込んでJPGとして保存
            img = cv2.imread(file)
            if img is not None:
                cv2.imwrite(str(filepath), img)
                self.import_log.insert(tk.END, f"インポート: {filepath}\n")
                next_num += 1
            else:
                self.import_log.insert(tk.END, f"エラー: {file} を読み込めませんでした\n")

        self.import_log.see(tk.END)

    # ==================== アノテーションタブ ====================
    def create_annotation_tab(self):
        """アノテーションタブを作成"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="アノテーション")

        # 左側：画像リスト
        left_frame = ttk.Frame(tab, width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)

        # フィルター
        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(fill=tk.X, pady=5)

        self.filter_var = tk.StringVar(value="all")
        ttk.Radiobutton(filter_frame, text="全て", variable=self.filter_var, value="all",
                       command=self.update_image_list).pack(side=tk.LEFT)
        ttk.Radiobutton(filter_frame, text="未ラベル", variable=self.filter_var, value="unlabeled",
                       command=self.update_image_list).pack(side=tk.LEFT)
        ttk.Radiobutton(filter_frame, text="ラベル済", variable=self.filter_var, value="labeled",
                       command=self.update_image_list).pack(side=tk.LEFT)

        # データセット選択
        dataset_frame = ttk.Frame(left_frame)
        dataset_frame.pack(fill=tk.X, pady=5)

        self.dataset_var = tk.StringVar(value="train")
        ttk.Radiobutton(dataset_frame, text="Train", variable=self.dataset_var, value="train",
                       command=self.update_image_list).pack(side=tk.LEFT)
        ttk.Radiobutton(dataset_frame, text="Val", variable=self.dataset_var, value="val",
                       command=self.update_image_list).pack(side=tk.LEFT)

        # 画像リスト
        ttk.Label(left_frame, text="画像一覧:").pack(anchor=tk.W)

        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.image_listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.config(yscrollcommand=scrollbar.set)

        ttk.Button(left_frame, text="リスト更新", command=self.update_image_list).pack(pady=5)

        # 中央：アノテーションキャンバス
        center_frame = ttk.Frame(tab)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.annotation_canvas = tk.Canvas(center_frame, bg='gray', width=800, height=600)
        self.annotation_canvas.pack(fill=tk.BOTH, expand=True)

        # マウスイベントバインド
        self.annotation_canvas.bind('<Button-1>', self.on_canvas_click)
        self.annotation_canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.annotation_canvas.bind('<ButtonRelease-1>', self.on_canvas_release)
        self.annotation_canvas.bind('<Button-3>', self.on_canvas_right_click)
        self.annotation_canvas.bind('<Motion>', self.on_canvas_motion)
        self.annotation_canvas.bind('<Leave>', self.on_canvas_leave)

        # 右側：コントロール
        right_frame = ttk.Frame(tab, width=200)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        right_frame.pack_propagate(False)

        # クラス選択
        ttk.Label(right_frame, text="クラス選択:").pack(anchor=tk.W, pady=5)

        self.class_listbox = tk.Listbox(right_frame, height=10)
        self.class_listbox.pack(fill=tk.X, pady=5)
        self.class_listbox.bind('<<ListboxSelect>>', self.on_class_select)
        self.update_class_listbox()

        # アノテーションリスト
        ttk.Label(right_frame, text="アノテーション:").pack(anchor=tk.W, pady=5)

        self.bbox_listbox = tk.Listbox(right_frame, height=10)
        self.bbox_listbox.pack(fill=tk.X, pady=5)
        self.bbox_listbox.bind('<<ListboxSelect>>', self.on_bbox_select)

        ttk.Button(right_frame, text="選択したBBox削除", command=self.delete_selected_bbox).pack(pady=5)
        ttk.Button(right_frame, text="全BBox削除", command=self.delete_all_bboxes).pack(pady=5)

        # 保存ボタン
        ttk.Button(right_frame, text="アノテーション保存", command=self.save_annotations).pack(pady=10)

        # 次の画像へボタン
        ttk.Button(right_frame, text="次の画像へ", command=self.go_to_next_image).pack(pady=5)

        # 操作説明
        ttk.Label(right_frame, text="操作方法:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=5)
        ttk.Label(right_frame, text="左ドラッグ: BBox描画", font=('', 8)).pack(anchor=tk.W)
        ttk.Label(right_frame, text="右クリック: BBox削除", font=('', 8)).pack(anchor=tk.W)

        # 初期リスト更新
        self.update_image_list()

    def update_class_listbox(self):
        """クラスリストボックスを更新"""
        self.class_listbox.delete(0, tk.END)
        for i, cls_name in enumerate(self.classes):
            self.class_listbox.insert(tk.END, f"{i}: {cls_name}")
        if self.classes:
            self.class_listbox.selection_set(0)
            self.selected_class_id = 0

    def on_class_select(self, event):
        """クラス選択時"""
        selection = self.class_listbox.curselection()
        if selection:
            self.selected_class_id = selection[0]

    def update_image_list(self):
        """画像リストを更新"""
        self.image_listbox.delete(0, tk.END)

        dataset = self.dataset_var.get()
        image_dir = self.dataset_path / "images" / dataset
        label_dir = self.dataset_path / "labels" / dataset

        if not image_dir.exists():
            return

        images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

        for img_path in images:
            label_path = label_dir / (img_path.stem + ".txt")
            has_label = label_path.exists() and label_path.stat().st_size > 0

            filter_val = self.filter_var.get()
            if filter_val == "all":
                pass
            elif filter_val == "unlabeled" and has_label:
                continue
            elif filter_val == "labeled" and not has_label:
                continue

            prefix = "[L] " if has_label else "[U] "
            self.image_listbox.insert(tk.END, prefix + img_path.name)

    def on_image_select(self, event):
        """画像選択時"""
        selection = self.image_listbox.curselection()
        if not selection:
            return

        item = self.image_listbox.get(selection[0])
        filename = item[4:]  # "[L] " or "[U] " を除去

        dataset = self.dataset_var.get()
        self.current_image_path = self.dataset_path / "images" / dataset / filename

        self.load_image()
        self.load_annotations()

    def load_image(self):
        """画像を読み込んでキャンバスに表示"""
        if not self.current_image_path or not self.current_image_path.exists():
            return

        # 画像読み込み
        img = cv2.imread(str(self.current_image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_image = img

        # キャンバスサイズに合わせてスケール
        canvas_w = self.annotation_canvas.winfo_width()
        canvas_h = self.annotation_canvas.winfo_height()
        img_h, img_w = img.shape[:2]

        self.scale_factor = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)

        img_resized = cv2.resize(img, (new_w, new_h))
        self.display_image = Image.fromarray(img_resized)
        self.photo = ImageTk.PhotoImage(self.display_image)

        # オフセット計算（中央配置用）
        self.img_offset_x = (canvas_w - new_w) // 2
        self.img_offset_y = (canvas_h - new_h) // 2
        self.display_w = new_w
        self.display_h = new_h

        self.redraw_canvas()

    def load_annotations(self):
        """アノテーションを読み込み"""
        self.bboxes = []

        if not self.current_image_path:
            return

        dataset = self.dataset_var.get()
        label_path = self.dataset_path / "labels" / dataset / (self.current_image_path.stem + ".txt")

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        self.bboxes.append((class_id, x_center, y_center, width, height))

        self.update_bbox_listbox()
        self.redraw_canvas()

    def update_bbox_listbox(self):
        """BBoxリストボックスを更新"""
        self.bbox_listbox.delete(0, tk.END)
        for i, (class_id, x, y, w, h) in enumerate(self.bboxes):
            cls_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
            self.bbox_listbox.insert(tk.END, f"{i}: {cls_name} ({x:.3f}, {y:.3f})")

    def redraw_canvas(self):
        """キャンバスを再描画"""
        self.annotation_canvas.delete("all")

        if hasattr(self, 'photo') and self.photo:
            self.annotation_canvas.create_image(
                self.img_offset_x, self.img_offset_y,
                image=self.photo, anchor=tk.NW
            )

        # BBoxを描画
        if self.current_image is not None:
            img_h, img_w = self.current_image.shape[:2]

            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']

            for i, (class_id, x_center, y_center, width, height) in enumerate(self.bboxes):
                # YOLO形式からピクセル座標に変換
                x1 = int((x_center - width/2) * img_w * self.scale_factor) + self.img_offset_x
                y1 = int((y_center - height/2) * img_h * self.scale_factor) + self.img_offset_y
                x2 = int((x_center + width/2) * img_w * self.scale_factor) + self.img_offset_x
                y2 = int((y_center + height/2) * img_h * self.scale_factor) + self.img_offset_y

                color = colors[class_id % len(colors)]
                self.annotation_canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)

                # クラスラベル
                cls_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
                self.annotation_canvas.create_text(x1+2, y1-10, text=cls_name, fill=color, anchor=tk.W)

    def on_canvas_motion(self, event):
        """マウス移動時にクロスヘアガイドを表示"""
        self.annotation_canvas.delete("guide")

        if self.current_image is None or self.drawing:
            return

        # 画像領域内のみガイドを表示
        if not (self.img_offset_x <= event.x <= self.img_offset_x + self.display_w and
                self.img_offset_y <= event.y <= self.img_offset_y + self.display_h):
            return

        # 垂直線
        self.annotation_canvas.create_line(
            event.x, self.img_offset_y, event.x, self.img_offset_y + self.display_h,
            fill='#00ff00', width=1, dash=(4, 4), tags="guide"
        )
        # 水平線
        self.annotation_canvas.create_line(
            self.img_offset_x, event.y, self.img_offset_x + self.display_w, event.y,
            fill='#00ff00', width=1, dash=(4, 4), tags="guide"
        )

    def on_canvas_leave(self, event):
        """キャンバスからマウスが離れた時にガイドを消す"""
        self.annotation_canvas.delete("guide")

    def on_canvas_click(self, event):
        """キャンバスクリック時"""
        if self.current_image is None:
            return

        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect = None

    def on_canvas_drag(self, event):
        """キャンバスドラッグ時"""
        if not self.drawing or self.current_image is None:
            return

        if self.current_rect:
            self.annotation_canvas.delete(self.current_rect)

        self.current_rect = self.annotation_canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='white', width=2, dash=(4, 4)
        )

    def on_canvas_release(self, event):
        """キャンバスリリース時"""
        if not self.drawing or self.current_image is None:
            return

        self.drawing = False

        if self.current_rect:
            self.annotation_canvas.delete(self.current_rect)
            self.current_rect = None

        # 座標を画像座標に変換
        img_h, img_w = self.current_image.shape[:2]

        # キャンバス座標から画像座標へ
        x1 = (min(self.start_x, event.x) - self.img_offset_x) / self.scale_factor
        y1 = (min(self.start_y, event.y) - self.img_offset_y) / self.scale_factor
        x2 = (max(self.start_x, event.x) - self.img_offset_x) / self.scale_factor
        y2 = (max(self.start_y, event.y) - self.img_offset_y) / self.scale_factor

        # 境界チェック
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        # 小さすぎる矩形は無視
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self.redraw_canvas()
            return

        # YOLO形式に変換
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        self.bboxes.append((self.selected_class_id, x_center, y_center, width, height))
        self.update_bbox_listbox()
        self.redraw_canvas()

    def on_canvas_right_click(self, event):
        """右クリックで最も近いBBoxを削除"""
        if self.current_image is None or not self.bboxes:
            return

        img_h, img_w = self.current_image.shape[:2]
        click_x = (event.x - self.img_offset_x) / self.scale_factor / img_w
        click_y = (event.y - self.img_offset_y) / self.scale_factor / img_h

        # クリック位置を含むBBoxを探す
        for i, (class_id, x_center, y_center, width, height) in enumerate(self.bboxes):
            x1 = x_center - width/2
            y1 = y_center - height/2
            x2 = x_center + width/2
            y2 = y_center + height/2

            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                del self.bboxes[i]
                self.update_bbox_listbox()
                self.redraw_canvas()
                return

    def on_bbox_select(self, event):
        """BBoxリスト選択時"""
        self.redraw_canvas()

        selection = self.bbox_listbox.curselection()
        if selection:
            idx = selection[0]
            if idx < len(self.bboxes):
                # 選択されたBBoxをハイライト
                class_id, x_center, y_center, width, height = self.bboxes[idx]
                img_h, img_w = self.current_image.shape[:2]

                x1 = int((x_center - width/2) * img_w * self.scale_factor) + self.img_offset_x
                y1 = int((y_center - height/2) * img_h * self.scale_factor) + self.img_offset_y
                x2 = int((x_center + width/2) * img_w * self.scale_factor) + self.img_offset_x
                y2 = int((y_center + height/2) * img_h * self.scale_factor) + self.img_offset_y

                self.annotation_canvas.create_rectangle(x1, y1, x2, y2, outline='white', width=3)

    def delete_selected_bbox(self):
        """選択したBBoxを削除"""
        selection = self.bbox_listbox.curselection()
        if selection:
            idx = selection[0]
            if idx < len(self.bboxes):
                del self.bboxes[idx]
                self.update_bbox_listbox()
                self.redraw_canvas()

    def delete_all_bboxes(self):
        """全BBoxを削除"""
        if messagebox.askyesno("確認", "全てのアノテーションを削除しますか？"):
            self.bboxes = []
            self.update_bbox_listbox()
            self.redraw_canvas()

    def save_annotations(self, silent=False):
        """アノテーションを保存"""
        if not self.current_image_path:
            if not silent:
                messagebox.showwarning("警告", "画像が選択されていません")
            return False

        dataset = self.dataset_var.get()
        label_dir = self.dataset_path / "labels" / dataset
        label_path = label_dir / (self.current_image_path.stem + ".txt")

        with open(label_path, 'w') as f:
            for class_id, x_center, y_center, width, height in self.bboxes:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        if not silent:
            messagebox.showinfo("完了", f"アノテーションを保存しました:\n{label_path}")
        self.update_image_list()
        return True

    def go_to_next_image(self):
        """アノテーションを保存して次の画像へ移動"""
        # 現在の選択位置を取得（save_annotationsでリストが更新される前に取得する必要がある）
        selection = self.image_listbox.curselection()
        if not selection:
            return

        current_idx = selection[0]
        total_items = self.image_listbox.size()

        # 現在のアノテーションを保存（メッセージなし）
        self.save_annotations(silent=True)

        # 次の画像があれば選択
        if current_idx < total_items - 1:
            next_idx = current_idx + 1
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(next_idx)
            self.image_listbox.see(next_idx)
            # 画像を読み込み
            self.on_image_select(None)

    # ==================== 学習設定タブ ====================
    def create_training_tab(self):
        """学習設定・実行タブを作成"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="学習")

        # 左側：設定
        left_frame = ttk.LabelFrame(tab, text="学習設定")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # モデルサイズ
        ttk.Label(left_frame, text="モデルサイズ:").pack(anchor=tk.W, padx=5, pady=2)
        self.model_size_var = tk.StringVar(value="yolov8n")
        model_combo = ttk.Combobox(left_frame, textvariable=self.model_size_var,
                                   values=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        model_combo.pack(fill=tk.X, padx=5, pady=2)

        # エポック数
        ttk.Label(left_frame, text="エポック数:").pack(anchor=tk.W, padx=5, pady=2)
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(left_frame, textvariable=self.epochs_var).pack(fill=tk.X, padx=5, pady=2)

        # バッチサイズ
        ttk.Label(left_frame, text="バッチサイズ:").pack(anchor=tk.W, padx=5, pady=2)
        self.batch_var = tk.StringVar(value="16")
        ttk.Entry(left_frame, textvariable=self.batch_var).pack(fill=tk.X, padx=5, pady=2)

        # 画像サイズ
        ttk.Label(left_frame, text="画像サイズ:").pack(anchor=tk.W, padx=5, pady=2)
        self.imgsz_var = tk.StringVar(value="640")
        ttk.Entry(left_frame, textvariable=self.imgsz_var).pack(fill=tk.X, padx=5, pady=2)

        # 事前学習モデル
        ttk.Label(left_frame, text="事前学習モデル:").pack(anchor=tk.W, padx=5, pady=2)
        self.pretrained_var = tk.StringVar(value="yolov8n.pt")
        pretrained_frame = ttk.Frame(left_frame)
        pretrained_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Entry(pretrained_frame, textvariable=self.pretrained_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(pretrained_frame, text="...", width=3, command=self.browse_pretrained).pack(side=tk.RIGHT)

        # プロジェクト名
        ttk.Label(left_frame, text="プロジェクト名:").pack(anchor=tk.W, padx=5, pady=2)
        self.project_var = tk.StringVar(value="model")
        ttk.Entry(left_frame, textvariable=self.project_var).pack(fill=tk.X, padx=5, pady=2)

        # 実験名
        ttk.Label(left_frame, text="実験名:").pack(anchor=tk.W, padx=5, pady=2)
        self.exp_name_var = tk.StringVar(value="exp")
        ttk.Entry(left_frame, textvariable=self.exp_name_var).pack(fill=tk.X, padx=5, pady=2)

        # デバイス
        ttk.Label(left_frame, text="デバイス:").pack(anchor=tk.W, padx=5, pady=2)
        self.device_var = tk.StringVar(value="cpu")
        device_combo = ttk.Combobox(left_frame, textvariable=self.device_var,
                                    values=["0", "cpu"])
        device_combo.pack(fill=tk.X, padx=5, pady=2)

        # ボタン
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)

        self.train_btn = ttk.Button(btn_frame, text="学習開始", command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        self.stop_train_btn = ttk.Button(btn_frame, text="学習停止", command=self.stop_training, state=tk.DISABLED)
        self.stop_train_btn.pack(side=tk.LEFT, padx=5)

        # 右側：ログ
        right_frame = ttk.LabelFrame(tab, text="学習ログ")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.training_log = tk.Text(right_frame, wrap=tk.WORD)
        self.training_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.training_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.training_log.config(yscrollcommand=scrollbar.set)

        # データセット情報表示
        info_frame = ttk.LabelFrame(left_frame, text="データセット情報")
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.dataset_info_label = ttk.Label(info_frame, text="")
        self.dataset_info_label.pack(padx=5, pady=5)

        ttk.Button(info_frame, text="情報更新", command=self.update_dataset_info).pack(pady=5)
        self.update_dataset_info()

    def browse_pretrained(self):
        """事前学習モデルを選択"""
        filepath = filedialog.askopenfilename(
            title="事前学習モデルを選択",
            filetypes=[("PyTorch Model", "*.pt")]
        )
        if filepath:
            self.pretrained_var.set(filepath)

    def update_dataset_info(self):
        """データセット情報を更新"""
        train_images = len(list((self.dataset_path / "images" / "train").glob("*.jpg")))
        train_images += len(list((self.dataset_path / "images" / "train").glob("*.png")))

        val_images = len(list((self.dataset_path / "images" / "val").glob("*.jpg")))
        val_images += len(list((self.dataset_path / "images" / "val").glob("*.png")))

        train_labels = len(list((self.dataset_path / "labels" / "train").glob("*.txt")))
        val_labels = len(list((self.dataset_path / "labels" / "val").glob("*.txt")))

        info = f"Train: {train_images}画像, {train_labels}ラベル\n"
        info += f"Val: {val_images}画像, {val_labels}ラベル\n"
        info += f"クラス数: {len(self.classes)}"

        self.dataset_info_label.config(text=info)

    def start_training(self):
        """学習を開始"""
        if not ULTRALYTICS_AVAILABLE:
            messagebox.showerror("エラー", "ultralyticsがインストールされていません")
            return

        # data.yamlを更新
        self.save_classes()

        self.training = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_train_btn.config(state=tk.NORMAL)

        self.training_log.delete(1.0, tk.END)
        self.training_log.insert(tk.END, "学習を開始します...\n")

        # 別スレッドで学習
        self.training_thread = threading.Thread(target=self.train_model, daemon=True)
        self.training_thread.start()

    def train_model(self):
        """モデル学習（別スレッド）"""
        try:
            model_size = self.model_size_var.get()
            pretrained = self.pretrained_var.get()

            # モデル読み込み
            if os.path.exists(pretrained):
                model = YOLO(pretrained)
                self.log_training(f"事前学習モデルを読み込みました: {pretrained}\n")
            else:
                model = YOLO(f"{model_size}.pt")
                self.log_training(f"モデルを読み込みました: {model_size}.pt\n")

            # 学習パラメータ
            params = {
                'data': str(self.data_yaml_path),
                'epochs': int(self.epochs_var.get()),
                'batch': int(self.batch_var.get()),
                'imgsz': int(self.imgsz_var.get()),
                'project': str(self.model_path),
                'name': self.exp_name_var.get(),
                'device': self.device_var.get() if self.device_var.get() != "cpu" else "cpu",
                'verbose': True
            }

            self.log_training(f"学習パラメータ: {params}\n")
            self.log_training("-" * 50 + "\n")

            # 学習実行
            results = model.train(**params)

            self.log_training("-" * 50 + "\n")
            self.log_training("学習が完了しました！\n")
            self.log_training(f"結果: {results}\n")

        except Exception as e:
            self.log_training(f"\nエラー: {e}\n")
        finally:
            self.training = False
            self.root.after(0, self.training_finished)

    def log_training(self, message):
        """学習ログを追加"""
        self.root.after(0, lambda: self._append_log(message))

    def _append_log(self, message):
        """ログを追加（メインスレッド）"""
        self.training_log.insert(tk.END, message)
        self.training_log.see(tk.END)

    def training_finished(self):
        """学習完了時"""
        self.train_btn.config(state=tk.NORMAL)
        self.stop_train_btn.config(state=tk.DISABLED)

    def stop_training(self):
        """学習を停止"""
        self.training = False
        self.log_training("\n学習を停止しています...\n")
        messagebox.showinfo("情報", "学習停止リクエストを送信しました。\n現在のエポック終了後に停止します。")

    # ==================== クラス管理タブ ====================
    def create_class_management_tab(self):
        """クラス管理タブを作成"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="クラス管理")

        # クラスリスト
        list_frame = ttk.LabelFrame(tab, text="クラス一覧")
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.class_mgmt_listbox = tk.Listbox(list_frame, height=20)
        self.class_mgmt_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.update_class_mgmt_list()

        # コントロール
        ctrl_frame = ttk.Frame(tab)
        ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # 新規クラス追加
        ttk.Label(ctrl_frame, text="クラス名:").pack(anchor=tk.W, pady=5)
        self.new_class_var = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self.new_class_var).pack(fill=tk.X, pady=5)

        ttk.Button(ctrl_frame, text="クラス追加", command=self.add_class).pack(fill=tk.X, pady=5)
        ttk.Button(ctrl_frame, text="選択したクラスを削除", command=self.delete_class).pack(fill=tk.X, pady=5)
        ttk.Button(ctrl_frame, text="クラス名を編集", command=self.edit_class).pack(fill=tk.X, pady=5)

        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(ctrl_frame, text="data.yaml保存", command=self.save_and_reload_classes).pack(fill=tk.X, pady=5)

        # 現在のdata.yaml内容表示
        yaml_frame = ttk.LabelFrame(ctrl_frame, text="data.yaml内容")
        yaml_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.yaml_text = tk.Text(yaml_frame, height=15, width=30)
        self.yaml_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.update_yaml_display()

    def update_class_mgmt_list(self):
        """クラス管理リストを更新"""
        self.class_mgmt_listbox.delete(0, tk.END)
        for i, cls_name in enumerate(self.classes):
            self.class_mgmt_listbox.insert(tk.END, f"{i}: {cls_name}")

    def update_yaml_display(self):
        """data.yaml表示を更新"""
        self.yaml_text.delete(1.0, tk.END)
        if self.data_yaml_path.exists():
            with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
                self.yaml_text.insert(tk.END, f.read())

    def add_class(self):
        """クラスを追加"""
        name = self.new_class_var.get().strip()
        if not name:
            messagebox.showwarning("警告", "クラス名を入力してください")
            return

        if name in self.classes:
            messagebox.showwarning("警告", "同じ名前のクラスが既に存在します")
            return

        self.classes.append(name)
        self.new_class_var.set("")
        self.update_class_mgmt_list()
        self.update_class_listbox()

    def delete_class(self):
        """選択したクラスを削除"""
        selection = self.class_mgmt_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "削除するクラスを選択してください")
            return

        idx = selection[0]
        cls_name = self.classes[idx]

        if messagebox.askyesno("確認", f"クラス '{cls_name}' を削除しますか？\n"
                               "このクラスを使用しているアノテーションのIDがずれる可能性があります。"):
            del self.classes[idx]
            self.update_class_mgmt_list()
            self.update_class_listbox()

    def edit_class(self):
        """クラス名を編集"""
        selection = self.class_mgmt_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "編集するクラスを選択してください")
            return

        idx = selection[0]
        old_name = self.classes[idx]

        # 簡易ダイアログ
        new_name = tk.simpledialog.askstring("クラス名編集", f"新しいクラス名:", initialvalue=old_name)
        if new_name and new_name.strip():
            self.classes[idx] = new_name.strip()
            self.update_class_mgmt_list()
            self.update_class_listbox()

    def save_and_reload_classes(self):
        """クラス情報を保存してリロード"""
        self.save_classes()
        self.update_yaml_display()
        self.update_class_listbox()
        messagebox.showinfo("完了", "data.yamlを保存しました")

    def on_closing(self):
        """アプリ終了時の処理"""
        self.stop_camera()
        self.root.destroy()


# simpledialogのインポート
import tkinter.simpledialog


def main():
    root = tk.Tk()
    app = YOLOGuiApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
