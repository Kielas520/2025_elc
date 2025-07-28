import cv2
import numpy as np

class ORBMatcher:
    def __init__(self, min_match_count=5, ratio_thresh=0.7):
        """
        专为小图像优化的ORB匹配器
        :param min_match_count: 降低最小匹配点要求
        :param ratio_thresh: 保持相同的ratio测试阈值
        """
        # 调整ORB参数适应小图像
        self.orb = cv2.ORB_create(
            nfeatures=30,       # 减少特征点数量
            scaleFactor=1.2,    # 降低金字塔缩放因子
            nlevels=3,          # 减少金字塔层数
            edgeThreshold=3,    # 减小边缘阈值
            firstLevel=0,       # 从原始图像开始
            WTA_K=2             # 保持默认2点产生描述符
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.templates = {}
        self.min_match_count = min_match_count
        self.ratio_thresh = ratio_thresh

    def add_template(self, name, template_path):
        """加载并预处理小模板图像"""
        template = cv2.imread(template_path, 0)
        if template is None:
            raise ValueError(f"无法加载模板: {template_path}")
        
        # 小图像专用预处理
        template = self._preprocess_small_image(template)
        
        kp, des = self.orb.detectAndCompute(template, None)
        if des is None or len(des) < 2:
            # 终极fallback：添加人工特征点
            template, kp, des = self._add_artificial_features(template, name)
            
        self.templates[name] = {'kp': kp, 'des': des, 'shape': template.shape[::-1]}
        print(f"添加模板 {name} 成功，特征点: {len(kp)}")

    def _preprocess_small_image(self, img):
        """小图像专用预处理链"""
        # 1. 保边锐化
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        
        # 2. 自适应直方图均衡
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
        img = clahe.apply(img)
        
        # 3. 边缘增强
        edges = cv2.Canny(img, 50, 150)
        img = cv2.addWeighted(img, 0.7, edges, 0.3, 0)
        
        return img

    def _add_artificial_features(self, img, name):
        """添加人工特征点作为fallback"""
        h, w = img.shape
        kp = []
        
        # 在图像四角和中心添加固定特征点
        points = [
            (5, 5), (w-5, 5), (w//2, h//2), 
            (5, h-5), (w-5, h-5)
        ]
        
        for pt in points:
            kp.append(cv2.KeyPoint(x=pt[0], y=pt[1], size=10))
        
        # 计算人工特征点的描述符
        _, des = self.orb.compute(img, kp)
        
        print(f"警告: 模板 {name} 使用 {len(kp)} 个人工特征点")
        return img, kp, des

    def match(self, img, show_result=False):
        """匹配流程与之前保持一致"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = self._preprocess_small_image(img)
        kp2, des2 = self.orb.detectAndCompute(img, None)
        
        best_match = None
        for name, template in self.templates.items():
            if des2 is None:
                continue
                
            matches = self.bf.knnMatch(template['des'], des2, k=2)
            good = []
            try:
                for m,n in matches:
                    if m.distance < self.ratio_thresh * n.distance:
                        good.append(m)
            except:
                continue
            
            if len(good) > self.min_match_count:
                # ... (保持原有的单应性矩阵计算逻辑)
                current_match = {
                    'name': name,
                    'confidence': len(good)/len(template['kp']),
                    'num_matches': len(good)
                }
                if best_match is None or current_match['confidence'] > best_match['confidence']:
                    best_match = current_match
        
        return best_match