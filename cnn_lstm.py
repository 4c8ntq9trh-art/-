# ventilation_diagnosis_complete.py
import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import networkx as nx
import joblib
from collections import Counter
import traceback


# ==================== 环境配置 ====================
class VentilationEnvironment:
    """矿井通风诊断系统环境配置"""

    @staticmethod
    def setup():
        """
        完全配置TensorFlow运行环境
        屏蔽所有不必要的信息提示，同时保持性能优化
        """
        # 1. 设置环境变量 - 在导入TensorFlow之前必须完成
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示ERROR级别信息
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # 启用oneDNN优化（保持性能）
        os.environ['OMP_NUM_THREADS'] = '1'  # 控制线程数

        # 2. 屏蔽所有Python警告
        warnings.filterwarnings('ignore')

        # 3. 配置日志系统 - 在导入TensorFlow之前设置
        logging.basicConfig(level=logging.INFO)
        for logger_name in ['tensorflow', 'h5py', 'matplotlib']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        # 4. 配置matplotlib中文字体支持
        VentilationEnvironment._setup_matplotlib()

        print("🔧 环境配置完成 - 已屏蔽TensorFlow信息提示，启用了CPU优化")

    @staticmethod
    def _setup_matplotlib():
        """配置matplotlib中文字体支持"""
        try:
            # 设置matplotlib参数以支持中文显示
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            plt.rcParams['figure.dpi'] = 100  # 设置图形分辨率
            plt.rcParams['savefig.dpi'] = 300  # 设置保存图像的分辨率
            plt.rcParams['font.size'] = 12  # 设置字体大小

            # 测试中文字体是否可用
            import matplotlib.font_manager as fm
            test_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            available_fonts = []

            for font in test_fonts:
                if any(font in f.name for f in fm.fontManager.ttflist):
                    available_fonts.append(font)

            if available_fonts:
                plt.rcParams['font.sans-serif'] = available_fonts + ['DejaVu Sans']
                print(f" 中文字体配置成功: 使用 {available_fonts[0]}")
            else:
                print("️ 未找到中文字体，使用默认字体")

        except Exception as e:
            print(f"️ 字体配置警告: {e}")
            # 设置回退字体配置
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False


# 应用环境配置（必须在导入TensorFlow之前）
VentilationEnvironment.setup()

# ==================== 导入TensorFlow和其他库 ====================
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.constraints import MaxNorm

# 设置随机种子保证可重复性
tf.random.set_seed(42)
np.random.seed(42)

print(f" TensorFlow版本: {tf.__version__}")


# ==================== 矿井拓扑结构类 ====================
class MineTopology:
    """矿井巷道拓扑结构管理类"""

    def __init__(self):
        self.tunnels = {}
        self.graph = nx.DiGraph()
        self.tunnel_sequence = []  # 拓扑排序结果
        self.resistance_matrix = None  # 风阻影响矩阵

    def initialize_standard_topology(self):
        """初始化标准矿井拓扑结构 - 使用e1,e2,e3...命名"""
        # 定义巷道及其连接关系 - 使用e1,e2,e3...命名
        tunnels = {
            'e1': {'type': '入口', 'level': 0, 'name': '主井口'},
            'e2': {'type': '主巷道', 'level': 1, 'name': '运输大巷1'},
            'e3': {'type': '主巷道', 'level': 1, 'name': '运输大巷2'},
            'e4': {'type': '连接巷道', 'level': 2, 'name': '采区上山'},
            'e5': {'type': '工作面', 'level': 3, 'name': '工作面巷道1'},
            'e6': {'type': '工作面', 'level': 3, 'name': '工作面巷道2'},
            'e7': {'type': '工作面', 'level': 3, 'name': '工作面巷道3'},
            'e8': {'type': '回风', 'level': 2, 'name': '回风巷1'},
            'e9': {'type': '回风', 'level': 2, 'name': '回风巷2'},
            'e10': {'type': '出口', 'level': 0, 'name': '回风井'}
        }

        # 定义连接关系 (从 -> 到)
        connections = [
            ('e1', 'e2'),
            ('e1', 'e3'),
            ('e2', 'e4'),
            ('e3', 'e4'),
            ('e4', 'e5'),
            ('e4', 'e6'),
            ('e4', 'e7'),
            ('e5', 'e8'),
            ('e6', 'e8'),
            ('e7', 'e9'),
            ('e8', 'e10'),
            ('e9', 'e10')
        ]

        self.tunnels = tunnels
        self.graph.add_nodes_from(tunnels.keys())
        self.graph.add_edges_from(connections)

        # 执行拓扑排序
        try:
            self.tunnel_sequence = list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            # 如果图有环，使用其他排序方法
            self.tunnel_sequence = list(self.tunnels.keys())

        # 计算风阻影响矩阵
        self._calculate_resistance_influence_matrix()

        print(" 矿井拓扑结构初始化完成")
        print(f" 巷道数量: {len(self.tunnels)}")
        print(f" 连接数量: {len(connections)}")
        print(f" 拓扑排序结果: {self.tunnel_sequence}")

        # 打印巷道详细信息
        print("\n 巷道详细信息:")
        for tunnel_id, info in self.tunnels.items():
            print(f"  {tunnel_id}: {info['name']} ({info['type']}, 层级: {info['level']})")

        return self.tunnels, self.graph

    def _calculate_resistance_influence_matrix(self):
        """计算风阻影响矩阵 - 表示巷道间风阻变化的相互影响"""
        n_tunnels = len(self.tunnels)
        tunnel_names = list(self.tunnels.keys())

        # 初始化影响矩阵
        influence_matrix = np.zeros((n_tunnels, n_tunnels))

        # 基于网络拓扑计算影响系数
        for i, tunnel_i in enumerate(tunnel_names):
            for j, tunnel_j in enumerate(tunnel_names):
                if i == j:
                    # 自身影响最大
                    influence_matrix[i, j] = 1.0
                else:
                    # 计算拓扑距离影响
                    try:
                        # 计算最短路径距离
                        distance = nx.shortest_path_length(self.graph, tunnel_i, tunnel_j)
                        # 距离越近，影响越大
                        influence_matrix[i, j] = 0.5 / distance
                    except:
                        # 如果不可达，影响为0
                        influence_matrix[i, j] = 0.0

        self.resistance_matrix = influence_matrix
        print(" 风阻影响矩阵计算完成")
        return influence_matrix

    def get_tunnel_features(self, tunnel_name):
        """获取巷道的特征向量"""
        tunnel_info = self.tunnels.get(tunnel_name, {})
        features = {
            'level': tunnel_info.get('level', 0),
            'is_entrance': 1 if tunnel_info.get('type') == '入口' else 0,
            'is_exit': 1 if tunnel_info.get('type') == '出口' else 0,
            'is_workface': 1 if tunnel_info.get('type') == '工作面' else 0,
            'is_main': 1 if tunnel_info.get('type') == '主巷道' else 0,
            'is_ventilation': 1 if tunnel_info.get('type') == '回风' else 0,
            'connectivity': self.graph.degree(tunnel_name) if tunnel_name in self.graph else 0
        }
        return features

    def calculate_wind_resistance(self, wind_speeds, pressures, cross_sections):
        """
        根据风速、风压和断面面积计算风阻
        风阻 R = ΔP / (ρ * v^2 * A^2)
        其中: ΔP - 风压差, ρ - 空气密度, v - 风速, A - 断面面积
        """
        air_density = 1.2  # 空气密度 kg/m³

        resistances = {}
        for tunnel in self.tunnels.keys():
            if tunnel in wind_speeds and tunnel in pressures and tunnel in cross_sections:
                v = wind_speeds[tunnel]
                P = pressures[tunnel]
                A = cross_sections[tunnel]

                if v > 0 and A > 0:
                    # 计算风阻
                    resistance = P / (air_density * v ** 2 * A ** 2)
                else:
                    resistance = 0.0

                resistances[tunnel] = resistance

        return resistances

    def simulate_resistance_effect(self, original_resistances, changed_tunnel, change_factor):
        """
        模拟一条巷道风阻变化对其他巷道风阻的影响
        """
        tunnel_names = list(self.tunnels.keys())
        changed_idx = tunnel_names.index(changed_tunnel)

        # 计算影响向量
        influence_vector = self.resistance_matrix[changed_idx, :]

        # 计算新的风阻值
        new_resistances = original_resistances.copy()
        for i, tunnel in enumerate(tunnel_names):
            if tunnel == changed_tunnel:
                # 故障巷道的风阻直接变化
                new_resistances[tunnel] *= change_factor
            else:
                # 其他巷道受影响的風阻变化
                influence = influence_vector[i]
                resistance_change = (change_factor - 1.0) * influence * 0.3  # 衰减系数
                new_resistances[tunnel] *= (1.0 + resistance_change)

        return new_resistances

    def visualize_topology(self):
        """可视化矿井拓扑结构"""
        plt.figure(figsize=(15, 10))

        # 使用层次布局
        pos = {}
        level_nodes = {}

        # 按层级分组节点
        for node, attrs in self.tunnels.items():
            level = attrs['level']
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)

        # 为每个层级的节点分配位置
        for level, nodes in level_nodes.items():
            n_nodes = len(nodes)
            for i, node in enumerate(nodes):
                pos[node] = (i - n_nodes / 2, -level)

        # 根据节点类型设置颜色
        node_colors = []
        node_labels = {}
        for node in self.graph.nodes():
            node_type = self.tunnels[node]['type']
            node_labels[node] = f"{node}\n{self.tunnels[node]['name']}"

            if node_type == '入口':
                node_colors.append('lightgreen')
            elif node_type == '出口':
                node_colors.append('lightcoral')
            elif node_type == '工作面':
                node_colors.append('lightblue')
            elif node_type == '主巷道':
                node_colors.append('yellow')
            else:
                node_colors.append('lightgray')

        # 绘制图形
        nx.draw(self.graph, pos,
                labels=node_labels,
                node_color=node_colors,
                node_size=2000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                edgecolors='black',
                linewidths=1)

        plt.title('矿井通风系统拓扑结构图 (e1-e10巷道编号)', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 显示图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='入口'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='主巷道'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='工作面'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='连接巷道'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='出口')
        ]
        plt.legend(handles=legend_elements, loc='upper left')

        # 先显示再保存
        plt.show()

        # 保存拓扑图
        save_path = "D:/Project_python/mine_topology.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f" 拓扑结构图已保存为 '{save_path}'")


# ==================== 数据处理器 - 基于风速计算风阻 ====================
class VentilationDataProcessor:
    """矿井通风数据预处理类 - 基于风速计算风阻的故障诊断"""

    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.tunnel_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False
        self.data_file_path = "D:/Project_python/wind_speed_resistance_data.xls"
        self.topology = MineTopology()
        self.tunnels, self.graph = self.topology.initialize_standard_topology()

    def load_data(self):
        """加载基于风速计算风阻的传感器数据"""
        try:
            if os.path.exists(self.data_file_path):
                print(f" 从绝对路径加载风速风阻数据: {self.data_file_path}")
                data = pd.read_excel(self.data_file_path)
                print(f" 数据加载成功，形状: {data.shape}")
                return data
            else:
                print(f"️ 数据文件不存在: {self.data_file_path}")
                print(" 生成基于风速计算风阻的示例数据")
                data = self._create_wind_speed_resistance_sample_data()

                # 确保目录存在
                os.makedirs(os.path.dirname(self.data_file_path), exist_ok=True)
                data.to_excel(self.data_file_path, index=False)
                print(f" 示例数据已保存到: {self.data_file_path}")
                return data

        except Exception as e:
            print(f" 数据加载失败: {e}")
            print(" 使用生成的示例数据")
            return self._create_wind_speed_resistance_sample_data()

    def _create_wind_speed_resistance_sample_data(self):
        """创建基于风速计算风阻的样本数据"""
        np.random.seed(42)
        n_samples = 12000
        tunnel_names = list(self.tunnels.keys())
        n_tunnels = len(tunnel_names)

        # 初始化数据存储
        all_data = []
        labels = []
        fault_tunnels = []

        # 定义巷道基本参数
        base_wind_speeds = {}
        base_pressures = {}
        cross_sections = {}

        # 为每个巷道设置基本参数
        for tunnel in tunnel_names:
            tunnel_info = self.topology.get_tunnel_features(tunnel)
            # 基础风速与巷道类型相关
            if tunnel_info['is_entrance'] or tunnel_info['is_exit']:
                base_wind_speeds[tunnel] = 8.0 + np.random.uniform(-1, 1)
            elif tunnel_info['is_main']:
                base_wind_speeds[tunnel] = 6.0 + np.random.uniform(-0.8, 0.8)
            elif tunnel_info['is_workface']:
                base_wind_speeds[tunnel] = 4.0 + np.random.uniform(-0.5, 0.5)
            else:
                base_wind_speeds[tunnel] = 5.0 + np.random.uniform(-0.6, 0.6)

            # 基础风压
            base_pressures[tunnel] = 1000 + tunnel_info['level'] * 50 + np.random.uniform(-20, 20)

            # 断面面积 (m²)
            if tunnel_info['is_main']:
                cross_sections[tunnel] = 12.0 + np.random.uniform(-1, 1)
            elif tunnel_info['is_workface']:
                cross_sections[tunnel] = 8.0 + np.random.uniform(-0.8, 0.8)
            else:
                cross_sections[tunnel] = 10.0 + np.random.uniform(-1, 1)

        # 正常状态的基础风阻
        base_resistances = self.topology.calculate_wind_resistance(
            base_wind_speeds, base_pressures, cross_sections
        )

        # 生成时间序列数据
        time = np.linspace(0, 200, n_samples)

        for i in range(n_samples):
            # 基础风速波动
            current_wind_speeds = base_wind_speeds.copy()
            current_pressures = base_pressures.copy()

            # 添加正常波动
            for tunnel in tunnel_names:
                # 周期性波动
                periodic = 0.3 * np.sin(2 * np.pi * 0.01 * time[i] + hash(tunnel) % 10)
                # 趋势性变化
                trend = 0.0001 * time[i]
                # 随机噪声
                noise = 0.1 * np.random.randn()

                current_wind_speeds[tunnel] = base_wind_speeds[tunnel] + periodic + trend + noise
                current_pressures[tunnel] = base_pressures[tunnel] + 10 * periodic + 5 * noise

            # 确定状态和故障
            if i < 6000:
                # 正常状态
                labels.append('正常')
                fault_tunnels.append('无故障')
                current_resistances = base_resistances.copy()
            else:
                # 故障状态
                labels.append('故障')
                # 随机选择故障巷道
                fault_tunnel = np.random.choice(tunnel_names)
                fault_tunnels.append(fault_tunnel)

                # 故障强度
                fault_intensity = np.random.uniform(1.5, 3.0)  # 风阻增加倍数

                # 模拟风阻变化及其对系统的影响
                current_resistances = self.topology.simulate_resistance_effect(
                    base_resistances, fault_tunnel, fault_intensity
                )

                # 根据新的风阻调整风速和风压（简化模拟）
                for tunnel in tunnel_names:
                    resistance_change_ratio = current_resistances[tunnel] / base_resistances[tunnel]
                    # 风阻增加会导致风速下降
                    current_wind_speeds[tunnel] /= np.sqrt(resistance_change_ratio)
                    # 风压相应调整
                    current_pressures[tunnel] *= (1 + 0.1 * (resistance_change_ratio - 1))

            # 计算当前时间步的特征
            sample_features = []

            # 1. 各巷道风速特征
            for tunnel in tunnel_names:
                sample_features.extend([
                    current_wind_speeds[tunnel],
                    current_pressures[tunnel],
                    cross_sections[tunnel]
                ])

            # 2. 计算并添加风阻特征
            calculated_resistances = self.topology.calculate_wind_resistance(
                current_wind_speeds, current_pressures, cross_sections
            )

            for tunnel in tunnel_names:
                sample_features.append(calculated_resistances.get(tunnel, 0.0))

            # 3. 添加风速变化率特征
            if i > 0:
                for tunnel in tunnel_names:
                    wind_speed_change = current_wind_speeds[tunnel] - all_data[i - 1][tunnel_names.index(tunnel) * 3]
                    sample_features.append(wind_speed_change)
            else:
                for _ in tunnel_names:
                    sample_features.append(0.0)

            # 4. 添加拓扑特征
            for tunnel in tunnel_names:
                tunnel_features = self.topology.get_tunnel_features(tunnel)
                sample_features.extend([
                    tunnel_features['level'],
                    tunnel_features['connectivity'],
                    tunnel_features['is_workface']
                ])

            all_data.append(sample_features)

        # 创建特征名称
        feature_names = []

        # 风速、风压、断面面积特征
        for tunnel in tunnel_names:
            tunnel_name = self.tunnels[tunnel]['name']
            feature_names.extend([
                f'{tunnel}_{tunnel_name}_风速',
                f'{tunnel}_{tunnel_name}_风压',
                f'{tunnel}_{tunnel_name}_断面面积'
            ])

        # 风阻特征
        for tunnel in tunnel_names:
            tunnel_name = self.tunnels[tunnel]['name']
            feature_names.append(f'{tunnel}_{tunnel_name}_风阻')

        # 风速变化率特征
        for tunnel in tunnel_names:
            tunnel_name = self.tunnels[tunnel]['name']
            feature_names.append(f'{tunnel}_{tunnel_name}_风速变化率')

        # 拓扑特征
        for tunnel in tunnel_names:
            feature_names.extend([
                f'{tunnel}_层级',
                f'{tunnel}_连通度',
                f'{tunnel}_是否工作面'
            ])

        # 创建DataFrame
        df = pd.DataFrame(all_data, columns=feature_names)
        df['状态'] = labels
        df['故障巷道'] = fault_tunnels
        df['时间戳'] = pd.date_range(start='2024-01-01', periods=n_samples, freq='min')

        print(f" 基于风速计算风阻的数据生成完成: {df.shape}")
        status_distribution = df['状态'].value_counts()
        fault_tunnel_distribution = df[df['状态'] == '故障']['故障巷道'].value_counts()

        print(f" 状态分布: {status_distribution.to_dict()}")
        print(f" 故障巷道分布: {fault_tunnel_distribution.to_dict()}")

        return df

    def preprocess_data(self, data, test_size=0.15, val_size=0.15):
        """数据预处理 - 基于风速风阻特征"""
        try:
            # 分离特征和标签
            feature_cols = [col for col in data.columns if col not in ['状态', '故障巷道', '时间戳']]
            self.feature_names = feature_cols

            X = data[feature_cols].values
            y_status = data['状态'].values
            y_tunnel = data['故障巷道'].values

            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)

            # 编码标签
            y_status_encoded = self.label_encoder.fit_transform(y_status)
            y_tunnel_encoded = self.tunnel_encoder.fit_transform(y_tunnel)

            self.is_fitted = True

            print(f" 数据预处理完成 - 特征: {X_scaled.shape}")
            print(f" 状态类别: {list(self.label_encoder.classes_)}")
            print(f" 巷道类别: {list(self.tunnel_encoder.classes_)}")

            # 计算类别权重用于参考
            status_class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_status_encoded),
                y=y_status_encoded
            )
            self.status_class_weights = dict(enumerate(status_class_weights))

            tunnel_class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_tunnel_encoded),
                y=y_tunnel_encoded
            )
            self.tunnel_class_weights = dict(enumerate(tunnel_class_weights))

            print(f"️ 状态类别权重: {self.status_class_weights}")
            print(f"️ 巷道类别权重: {self.tunnel_class_weights}")

            return X_scaled, y_status_encoded, y_tunnel_encoded

        except Exception as e:
            print(f" 数据预处理失败: {e}")
            raise

    def create_sequences(self, X, y_status, y_tunnel, step_size=5):
        """创建时间序列数据"""
        if not self.is_fitted:
            raise ValueError("数据处理器尚未拟合，请先调用preprocess_data方法")

        sequences = []
        status_labels = []
        tunnel_labels = []

        for i in range(0, len(X) - self.sequence_length + 1, step_size):
            sequences.append(X[i:(i + self.sequence_length)])
            status_labels.append(y_status[i + self.sequence_length - 1])
            tunnel_labels.append(y_tunnel[i + self.sequence_length - 1])

        sequences = np.array(sequences)
        status_labels = np.array(status_labels)
        tunnel_labels = np.array(tunnel_labels)

        print(f" 序列数据创建完成 - 序列形状: {sequences.shape}")
        print(f" 状态标签形状: {status_labels.shape}")
        print(f" 巷道标签形状: {tunnel_labels.shape}")

        return sequences, status_labels, tunnel_labels

    def save_preprocessor(self, file_path=None):
        """保存预处理器状态"""
        if file_path is None:
            preprocessor_dir = "D:/Project_python"
            os.makedirs(preprocessor_dir, exist_ok=True)
            file_path = os.path.join(preprocessor_dir, "wind_speed_resistance_preprocessor.pkl")

        preprocessor_state = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'tunnel_encoder': self.tunnel_encoder,
            'sequence_length': self.sequence_length,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'status_class_weights': getattr(self, 'status_class_weights', None),
            'tunnel_class_weights': getattr(self, 'tunnel_class_weights', None),
            'data_file_path': self.data_file_path
        }
        joblib.dump(preprocessor_state, file_path)
        print(f" 预处理器状态已保存到: {file_path}")

    def load_preprocessor(self, file_path=None):
        """加载预处理器状态"""
        if file_path is None:
            file_path = "D:/Project_python/wind_speed_resistance_preprocessor.pkl"

        try:
            preprocessor_state = joblib.load(file_path)
            self.scaler = preprocessor_state['scaler']
            self.label_encoder = preprocessor_state['label_encoder']
            self.tunnel_encoder = preprocessor_state['tunnel_encoder']
            self.sequence_length = preprocessor_state['sequence_length']
            self.feature_names = preprocessor_state['feature_names']
            self.is_fitted = preprocessor_state['is_fitted']
            self.status_class_weights = preprocessor_state.get('status_class_weights', None)
            self.tunnel_class_weights = preprocessor_state.get('tunnel_class_weights', None)
            self.data_file_path = preprocessor_state.get('data_file_path', self.data_file_path)
            print(f" 预处理器状态已从 {file_path} 加载")
        except Exception as e:
            print(f" 预处理器加载失败: {e}")
            print(" 使用新的预处理器")
            self.topology = MineTopology()
            self.tunnels, self.graph = self.topology.initialize_standard_topology()


# ==================== 增强的多任务学习模型 ====================
class EnhancedMultiTaskCNNLSTMModel:
    """增强的多任务学习CNN-LSTM混合模型 - 专门处理风阻传播效应"""

    def __init__(self, input_shape, num_status_classes=2, num_tunnel_classes=11,
                 model_name="enhanced_ventilation_model"):
        self.input_shape = input_shape
        self.num_status_classes = num_status_classes
        self.num_tunnel_classes = num_tunnel_classes
        self.model_name = model_name
        self.model = None
        self.history = None
        self.label_encoder = None
        self.tunnel_encoder = None
        self.lr_history = []

    def build_enhanced_model(self, learning_rate=0.001):
        """构建增强的多任务学习模型，专门处理风阻传播效应"""
        try:
            # 输入层
            inputs = Input(shape=self.input_shape, name='input')

            # 增强的特征提取层 - 使用更深的网络捕获复杂模式
            # 第一卷积块
            x = Conv1D(filters=64, kernel_size=7, padding='same', activation='relu',
                       kernel_regularizer=l2(0.002))(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = SpatialDropout1D(0.2)(x)

            # 第二卷积块
            x = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu',
                       kernel_regularizer=l2(0.002))(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = SpatialDropout1D(0.25)(x)

            # 第三卷积块
            x = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu',
                       kernel_regularizer=l2(0.001))(x)
            x = BatchNormalization()(x)
            x = SpatialDropout1D(0.3)(x)

            # 第四卷积块 - 捕获更细微的模式
            x = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu',
                       kernel_regularizer=l2(0.001))(x)
            x = BatchNormalization()(x)
            x = SpatialDropout1D(0.3)(x)

            # 双向LSTM层 - 增强时序特征提取
            x = Bidirectional(LSTM(units=256, return_sequences=True,
                                   kernel_regularizer=l2(0.001),
                                   recurrent_regularizer=l2(0.001)))(x)
            x = Dropout(0.4)(x)

            x = Bidirectional(LSTM(units=128, return_sequences=True,
                                   kernel_regularizer=l2(0.001),
                                   recurrent_regularizer=l2(0.001)))(x)
            x = Dropout(0.4)(x)

            x = Bidirectional(LSTM(units=64, return_sequences=False,
                                   kernel_regularizer=l2(0.001),
                                   recurrent_regularizer=l2(0.001)))(x)
            x = Dropout(0.4)(x)

            # 注意力机制
            attention = Dense(64, activation='tanh')(x)
            attention_weights = Dense(1, activation='softmax')(attention)
            weighted_features = Multiply()([x, attention_weights])

            # 共享的密集层
            shared_features = Dense(units=128, activation='relu', name='shared_features')(weighted_features)
            shared_features = Dropout(0.3)(shared_features)
            shared_features = Dense(units=64, activation='relu')(shared_features)
            shared_features = Dropout(0.2)(shared_features)

            # 任务1：故障诊断（二分类）
            status_branch = Dense(units=32, activation='relu', name='status_branch')(shared_features)
            status_branch = Dropout(0.2)(status_branch)
            status_output = Dense(units=self.num_status_classes, activation='softmax', name='status_output')(
                status_branch)

            # 任务2：故障巷道定位（多分类）
            tunnel_branch = Dense(units=64, activation='relu', name='tunnel_branch')(shared_features)
            tunnel_branch = Dropout(0.3)(tunnel_branch)
            tunnel_branch = Dense(units=32, activation='relu')(tunnel_branch)
            tunnel_branch = Dropout(0.2)(tunnel_branch)
            tunnel_output = Dense(units=self.num_tunnel_classes, activation='softmax', name='tunnel_output')(
                tunnel_branch)

            # 创建多输出模型
            model = tf.keras.Model(
                inputs=inputs,
                outputs=[status_output, tunnel_output],
                name=self.model_name
            )

            # 编译模型
            optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)

            model.compile(
                optimizer=optimizer,
                loss={
                    'status_output': 'sparse_categorical_crossentropy',
                    'tunnel_output': 'sparse_categorical_crossentropy'
                },
                loss_weights={
                    'status_output': 0.6,  # 故障诊断任务权重
                    'tunnel_output': 0.4  # 巷道定位任务权重
                },
                metrics={
                    'status_output': ['accuracy', 'precision', 'recall'],
                    'tunnel_output': ['accuracy']
                }
            )

            self.model = model
            print(" 增强的多任务学习模型构建完成")
            print(f" 输入形状: {self.input_shape}")
            print(f" 任务1 - 故障诊断: {self.num_status_classes}类")
            print(f" 任务2 - 巷道定位: {self.num_tunnel_classes}类")
            print(f" 模型总参数: {model.count_params():,}")

            return model

        except Exception as e:
            print(f" 模型构建失败: {e}")
            raise

    def train_with_enhanced_validation(self, X_train, y_status_train, y_tunnel_train,
                                       X_val, y_status_val, y_tunnel_val,
                                       epochs=150, batch_size=32):
        """使用增强的训练策略训练模型"""
        try:
            checkpoint_dir = "D:/Project_python/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

            self.lr_history = []

            # 增强的回调函数
            callbacks = [
                EarlyStopping(
                    monitor='val_status_output_accuracy',
                    patience=25,
                    restore_best_weights=True,
                    verbose=1,
                    mode='max',
                    min_delta=0.001
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=12,
                    min_lr=1e-7,
                    verbose=1,
                    mode='min',
                    min_delta=0.001
                ),
                ModelCheckpoint(
                    filepath=os.path.join(checkpoint_dir, f"{self.model_name}_best.keras"),
                    monitor='val_status_output_accuracy',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                    mode='max'
                ),
                # 学习率调度器
                LearningRateScheduler(self._step_decay_schedule)
            ]

            print(f" 开始增强的多任务学习训练")
            print(f" 训练参数: 轮数={epochs}, 批次大小={batch_size}")

            # 训练数据准备
            train_data = {
                'status_output': y_status_train,
                'tunnel_output': y_tunnel_train
            }

            val_data = {
                'status_output': y_status_val,
                'tunnel_output': y_tunnel_val
            }

            # 训练模型
            self.history = self.model.fit(
                X_train, train_data,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, val_data),
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )

            print(" 增强的多任务模型训练完成")
            return self.history

        except Exception as e:
            print(f" 模型训练失败: {e}")
            raise

    def _step_decay_schedule(self, epoch, lr):
        """学习率衰减策略"""
        if epoch > 0 and epoch % 30 == 0:
            new_lr = lr * 0.5
            print(f" 学习率从 {lr:.6f} 降低到 {new_lr:.6f}")
            return new_lr
        return lr

    def comprehensive_evaluate(self, X_test, y_status_test, y_tunnel_test):
        """全面评估增强模型性能"""
        try:
            # 预测
            predictions = self.model.predict(X_test, verbose=0)
            status_pred_proba, tunnel_pred_proba = predictions
            status_pred = np.argmax(status_pred_proba, axis=1)
            tunnel_pred = np.argmax(tunnel_pred_proba, axis=1)

            # 故障诊断任务评估
            status_accuracy = accuracy_score(y_status_test, status_pred)
            status_precision = precision_score(y_status_test, status_pred, average='binary', zero_division=0)
            status_recall = recall_score(y_status_test, status_pred, average='binary', zero_division=0)
            status_f1 = f1_score(y_status_test, status_pred, average='binary', zero_division=0)

            print(f" 故障诊断任务性能:")
            print(f"   准确率: {status_accuracy:.4f}")
            print(f"   精确率: {status_precision:.4f}")
            print(f"   召回率: {status_recall:.4f}")
            print(f"  ️ F1分数: {status_f1:.4f}")

            # 巷道定位任务评估（只评估故障样本）
            fault_mask = y_status_test == 1
            tunnel_accuracy = 0
            if np.any(fault_mask):
                tunnel_accuracy = accuracy_score(y_tunnel_test[fault_mask], tunnel_pred[fault_mask])
                print(f" 巷道定位任务性能 (仅故障样本):")
                print(f"   准确率: {tunnel_accuracy:.4f}")

            # 详细分类报告
            if self.label_encoder:
                status_names = self.label_encoder.classes_
            else:
                status_names = ['正常', '故障']

            print("\n 故障诊断详细报告:")
            print(classification_report(y_status_test, status_pred, target_names=status_names, digits=4))

            # 绘制增强的结果可视化
            self._plot_enhanced_results(y_status_test, status_pred, y_tunnel_test, tunnel_pred)

            return (status_pred, tunnel_pred), {
                'status_accuracy': status_accuracy,
                'status_precision': status_precision,
                'status_recall': status_recall,
                'status_f1': status_f1,
                'tunnel_accuracy': tunnel_accuracy if np.any(fault_mask) else 0
            }

        except Exception as e:
            print(f"❌ 模型评估失败: {e}")
            raise

    def _plot_enhanced_results(self, y_status_true, y_status_pred, y_tunnel_true, y_tunnel_pred):
        """绘制增强的多任务学习结果"""
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()

        # 1. 故障诊断混淆矩阵
        cm_status = confusion_matrix(y_status_true, y_status_pred)
        sns.heatmap(cm_status, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['正常', '故障'], yticklabels=['正常', '故障'])
        axes[0].set_title('故障诊断混淆矩阵', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('预测标签')
        axes[0].set_ylabel('真实标签')

        # 2. 巷道定位混淆矩阵（仅故障样本）
        fault_mask = y_status_true == 1
        if np.any(fault_mask) and self.tunnel_encoder:
            cm_tunnel = confusion_matrix(y_tunnel_true[fault_mask], y_tunnel_pred[fault_mask])
            tunnel_names = self.tunnel_encoder.classes_
            sns.heatmap(cm_tunnel, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                        xticklabels=tunnel_names, yticklabels=tunnel_names)
            axes[1].set_title('巷道定位混淆矩阵 (仅故障样本)', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('预测巷道')
            axes[1].set_ylabel('真实巷道')
            plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
            plt.setp(axes[1].get_yticklabels(), rotation=0)
        else:
            axes[1].text(0.5, 0.5, '无故障样本数据', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('巷道定位混淆矩阵', fontsize=14, fontweight='bold')

        # 3. 训练历史 - 准确率
        if self.history is not None:
            epochs = range(1, len(self.history.history['status_output_accuracy']) + 1)
            axes[2].plot(epochs, self.history.history['status_output_accuracy'],
                         label='故障诊断训练准确率', linewidth=2)
            axes[2].plot(epochs, self.history.history['val_status_output_accuracy'],
                         label='故障诊断验证准确率', linewidth=2)
            if 'tunnel_output_accuracy' in self.history.history:
                axes[2].plot(epochs, self.history.history['tunnel_output_accuracy'],
                             label='巷道定位训练准确率', linewidth=2, linestyle='--')
                axes[2].plot(epochs, self.history.history['val_tunnel_output_accuracy'],
                             label='巷道定位验证准确率', linewidth=2, linestyle='--')
            axes[2].set_title('训练历史 - 准确率', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('训练轮数')
            axes[2].set_ylabel('准确率')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        # 4. 训练历史 - 损失
        if self.history is not None:
            axes[3].plot(epochs, self.history.history['loss'],
                         label='总训练损失', linewidth=2)
            axes[3].plot(epochs, self.history.history['val_loss'],
                         label='总验证损失', linewidth=2)
            axes[3].set_title('训练历史 - 损失', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('训练轮数')
            axes[3].set_ylabel('损失值')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        # 5. 故障检测概率分布
        fault_indices = np.where(y_status_true == 1)[0]
        normal_indices = np.where(y_status_true == 0)[0]

        if len(fault_indices) > 0 and len(normal_indices) > 0:
            predictions = self.model.predict(X_test, verbose=0)
            status_pred_proba, _ = predictions
            fault_probs = status_pred_proba[:, 1]

            axes[4].hist(fault_probs[normal_indices], bins=30, alpha=0.7, label='正常样本', color='green')
            axes[4].hist(fault_probs[fault_indices], bins=30, alpha=0.7, label='故障样本', color='red')
            axes[4].set_title('故障检测概率分布', fontsize=14, fontweight='bold')
            axes[4].set_xlabel('故障概率')
            axes[4].set_ylabel('样本数量')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)

        # 6. 特征重要性分析（简化版）
        if hasattr(self, 'feature_importance'):
            feature_names = getattr(self, 'feature_names', [f'Feature_{i}' for i in range(10)])
            top_features = min(10, len(feature_names))
            indices = np.argsort(self.feature_importance)[-top_features:]

            axes[5].barh(range(top_features), self.feature_importance[indices])
            axes[5].set_yticks(range(top_features))
            axes[5].set_yticklabels([feature_names[i] for i in indices])
            axes[5].set_title('Top 10 重要特征', fontsize=14, fontweight='bold')
            axes[5].set_xlabel('特征重要性')

        plt.tight_layout()
        plt.show()

        # 保存结果图
        save_path = "D:/Project_python/enhanced_multi_task_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f" 增强的多任务结果图已保存为 '{save_path}'")

    def save_model(self, file_path=None):
        """保存模型"""
        if file_path is None:
            model_dir = "D:/Project_python"
            os.makedirs(model_dir, exist_ok=True)
            file_path = os.path.join(model_dir, "enhanced_ventilation_multi_task_model.keras")

        self.model.save(file_path)
        print(f" 增强的多任务模型已保存到: {file_path}")

    def load_model(self, file_path=None):
        """加载模型"""
        if file_path is None:
            file_path = "D:/Project_python/enhanced_ventilation_multi_task_model.keras"

        self.model = load_model(file_path)
        print(f" 模型已从 {file_path} 加载")


# ==================== 增强的实时诊断类 ====================
class EnhancedRealTimeDiagnosis:
    """增强的实时故障诊断类 - 专门处理风阻传播效应"""

    def __init__(self, model, data_processor):
        self.model = model
        self.data_processor = data_processor
        self.data_buffer = []
        self.confidence_history = []
        self.fault_probability_history = []
        self.recent_tunnel_predictions = []
        self.wind_speed_trends = {}
        self.resistance_anomalies = {}

    def add_data(self, new_data):
        """添加新数据到缓冲区"""
        self.data_buffer.append(new_data)

        if len(self.data_buffer) > self.data_processor.sequence_length * 2:
            self.data_buffer = self.data_buffer[-self.data_processor.sequence_length * 2:]

    def analyze_wind_speed_trends(self, sequence_data):
        """分析风速趋势特征"""
        trends = {}

        # 计算各巷道的风速变化趋势
        for i in range(sequence_data.shape[1]):
            # 假设风速数据在特定位置
            wind_speed_data = sequence_data[:, i]
            if len(wind_speed_data) > 1:
                # 计算趋势斜率
                x = np.arange(len(wind_speed_data))
                slope, _ = np.polyfit(x, wind_speed_data, 1)
                trends[f'feature_{i}_trend'] = slope

        return trends

    def detect_resistance_anomalies(self, sequence_data):
        """检测风阻异常模式"""
        anomalies = {}

        # 计算风阻特征的统计异常
        resistance_features = [i for i, name in enumerate(self.data_processor.feature_names)
                               if '风阻' in name]

        for feature_idx in resistance_features:
            feature_data = sequence_data[:, feature_idx]
            mean_val = np.mean(feature_data)
            std_val = np.std(feature_data)

            # 检测异常点
            z_scores = np.abs((feature_data - mean_val) / (std_val + 1e-8))
            anomaly_count = np.sum(z_scores > 2.0)
            anomalies[f'resistance_{feature_idx}_anomalies'] = anomaly_count / len(feature_data)

        return anomalies

    def diagnose_with_enhanced_location(self, confidence_threshold=0.7):
        """包含增强巷道定位的实时诊断"""
        if len(self.data_buffer) < self.data_processor.sequence_length:
            return "数据不足", 0.0, "未知", 0.0, {"error": "数据不足"}

        try:
            sequence_data = np.array(self.data_buffer[-self.data_processor.sequence_length:])
            sequence_scaled = self.data_processor.scaler.transform(sequence_data)
            sequence_reshaped = sequence_scaled.reshape(1, self.data_processor.sequence_length, -1)

            # 分析辅助特征
            wind_trends = self.analyze_wind_speed_trends(sequence_data)
            resistance_anomalies = self.detect_resistance_anomalies(sequence_data)

            # 多任务预测
            predictions = self.model.model.predict(sequence_reshaped, verbose=0)
            status_pred_proba, tunnel_pred_proba = predictions

            fault_probability = status_pred_proba[0][1]
            confidence = max(fault_probability, 1 - fault_probability)

            # 故障诊断结果
            if fault_probability > 0.5:
                diagnosis_result = "故障"
                # 巷道定位
                tunnel_pred = np.argmax(tunnel_pred_proba[0])
                tunnel_confidence = tunnel_pred_proba[0][tunnel_pred]

                if self.data_processor.tunnel_encoder:
                    predicted_tunnel = self.data_processor.tunnel_encoder.inverse_transform([tunnel_pred])[0]
                    # 如果是"无故障"，重新选择第二可能的巷道
                    if predicted_tunnel == '无故障':
                        sorted_indices = np.argsort(tunnel_pred_proba[0])[::-1]
                        for idx in sorted_indices[1:]:
                            alternative_tunnel = self.data_processor.tunnel_encoder.inverse_transform([idx])[0]
                            if alternative_tunnel != '无故障':
                                predicted_tunnel = alternative_tunnel
                                tunnel_confidence = tunnel_pred_proba[0][idx]
                                break
                else:
                    predicted_tunnel = f"巷道_{tunnel_pred}"
            else:
                diagnosis_result = "正常"
                predicted_tunnel = "无故障"
                tunnel_confidence = 1 - fault_probability

            # 更新历史记录
            self.confidence_history.append(confidence)
            self.fault_probability_history.append(fault_probability)
            if diagnosis_result == "故障":
                self.recent_tunnel_predictions.append(predicted_tunnel)

            if len(self.confidence_history) > 10:
                self.confidence_history.pop(0)
                self.fault_probability_history.pop(0)
            if len(self.recent_tunnel_predictions) > 5:
                self.recent_tunnel_predictions.pop(0)

            # 分析结果稳定性
            confidence_stability = np.std(self.confidence_history) if len(self.confidence_history) > 1 else 0

            # 巷道定位一致性检查
            tunnel_consistency = 0
            if len(self.recent_tunnel_predictions) >= 3:
                tunnel_counts = Counter(self.recent_tunnel_predictions)
                most_common_tunnel, count = tunnel_counts.most_common(1)[0]
                tunnel_consistency = count / len(self.recent_tunnel_predictions)

            # 生成增强的诊断报告
            warnings = []
            recommendations = []

            if confidence < confidence_threshold:
                warnings.append(f"⚠️ 诊断置信度较低 ({confidence:.3f})")

            if confidence_stability > 0.1:
                warnings.append(f" 置信度波动较大 ({confidence_stability:.3f})")

            if diagnosis_result == "故障" and tunnel_consistency < 0.6:
                warnings.append(f" 巷道定位不一致 ({tunnel_consistency:.2f})")
                recommendations.append("建议检查相邻巷道的风阻情况")

            # 基于风阻异常的分析
            high_anomaly_features = [k for k, v in resistance_anomalies.items() if v > 0.3]
            if high_anomaly_features and diagnosis_result == "故障":
                warnings.append(f" 检测到风阻异常特征: {len(high_anomaly_features)}个")
                recommendations.append("风阻异常可能表明局部阻塞或变形")

            # 风速趋势分析
            negative_trends = [k for k, v in wind_trends.items() if v < -0.1]
            if negative_trends and diagnosis_result == "故障":
                warnings.append(f" 检测到风速下降趋势: {len(negative_trends)}个特征")
                recommendations.append("风速下降可能表明风阻增加")

            print(f" 实时诊断结果: {diagnosis_result}")
            print(f" 故障概率: {fault_probability:.4f}")
            print(f" 诊断置信度: {confidence:.4f}")
            if diagnosis_result == "故障":
                print(f" 预测故障巷道: {predicted_tunnel}")
                print(f" 巷道定位置信度: {tunnel_confidence:.4f}")
                print(f" 巷道定位一致性: {tunnel_consistency:.4f}")

            if warnings:
                print(" 警告信息:")
                for warning in warnings:
                    print(f"  {warning}")

            if recommendations:
                print(" 处理建议:")
                for recommendation in recommendations:
                    print(f"  {recommendation}")

            details = {
                'fault_probability': fault_probability,
                'confidence_stability': confidence_stability,
                'tunnel_consistency': tunnel_consistency,
                'wind_trends': wind_trends,
                'resistance_anomalies': resistance_anomalies,
                'warnings': warnings,
                'recommendations': recommendations,
                'recent_tunnel_predictions': self.recent_tunnel_predictions.copy(),
                'buffer_size': len(self.data_buffer)
            }

            return diagnosis_result, confidence, predicted_tunnel, tunnel_confidence, details

        except Exception as e:
            print(f"❌ 实时诊断失败: {e}")
            return "诊断失败", 0.0, "未知", 0.0, {"error": str(e)}


# ==================== 主程序 ====================
def main():
    """主函数 - 执行基于风速计算风阻的矿井通风故障诊断"""
    print("\n" + "=" * 70)
    print(" 基于风速计算风阻的矿井通风故障诊断系统")
    print("=" * 70)
    print(f" 工作目录: D:/Project_python/")
    print(" 系统特性:")
    print("  - 基于风速、风压、断面面积计算风阻")
    print("  - 考虑风阻变化的传播效应")
    print("  - 增强的CNN+LSTM多任务学习模型")
    print("  - 实时风阻异常检测和趋势分析")
    print("  - 可视化拓扑结构和诊断结果")

    try:
        # 1. 初始化拓扑结构
        print("\n  步骤1: 初始化矿井拓扑结构")
        topology = MineTopology()
        tunnels, graph = topology.initialize_standard_topology()
        topology.visualize_topology()

        # 2. 数据准备
        print("\n 步骤2: 准备基于风速计算风阻的数据")
        processor = VentilationDataProcessor(sequence_length=60)
        data = processor.load_data()

        # 数据预处理
        X, y_status, y_tunnel = processor.preprocess_data(data)

        # 创建序列数据
        X_seq, y_status_seq, y_tunnel_seq = processor.create_sequences(X, y_status, y_tunnel, step_size=10)

        # 数据分割
        X_train, X_test, y_status_train, y_status_test, y_tunnel_train, y_tunnel_test = train_test_split(
            X_seq, y_status_seq, y_tunnel_seq, test_size=0.15, random_state=42, stratify=y_status_seq
        )
        X_train, X_val, y_status_train, y_status_val, y_tunnel_train, y_tunnel_val = train_test_split(
            X_train, y_status_train, y_tunnel_train, test_size=0.15, random_state=42, stratify=y_status_train
        )

        print(f" 数据分割完成:")
        print(f"  训练集: {X_train.shape}")
        print(f"  验证集: {X_val.shape}")
        print(f"  测试集: {X_test.shape}")

        # 3. 增强的多任务模型构建
        print("\n  步骤3: 构建增强的多任务学习模型")
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_status_classes = 2
        num_tunnel_classes = len(processor.tunnel_encoder.classes_)

        model_builder = EnhancedMultiTaskCNNLSTMModel(input_shape, num_status_classes, num_tunnel_classes)
        model_builder.label_encoder = processor.label_encoder
        model_builder.tunnel_encoder = processor.tunnel_encoder
        model = model_builder.build_enhanced_model(learning_rate=0.001)

        # 4. 模型训练
        print("\n 步骤4: 增强的多任务模型训练")
        history = model_builder.train_with_enhanced_validation(
            X_train, y_status_train, y_tunnel_train,
            X_val, y_status_val, y_tunnel_val,
            epochs=150,
            batch_size=32
        )

        # 5. 模型评估
        print("\n 步骤5: 增强的多任务模型评估")
        predictions, metrics = model_builder.comprehensive_evaluate(X_test, y_status_test, y_tunnel_test)

        # 6. 保存模型
        print("\n 步骤6: 保存模型和预处理器")
        model_builder.save_model()
        processor.save_preprocessor()

        # 7. 实时诊断演示
        print("\n 步骤7: 增强的实时诊断演示")
        real_time_diagnoser = EnhancedRealTimeDiagnosis(model_builder, processor)

        # 使用测试数据进行演示
        demo_samples = X_test[:30]
        for i, sample in enumerate(demo_samples):
            for time_point in sample[-5:]:
                real_time_diagnoser.add_data(time_point)

            # 每5个样本进行一次诊断
            if i % 5 == 4:
                print(f"\n--- 增强诊断测试 {i // 5 + 1} ---")
                diagnosis, confidence, tunnel, tunnel_confidence, details = real_time_diagnoser.diagnose_with_enhanced_location()

                if diagnosis == "故障":
                    print(f" 检测到故障！位置: {tunnel} (置信度: {tunnel_confidence:.3f})")
                    if details.get('resistance_anomalies'):
                        print(
                            f" 风阻异常检测: {sum(1 for v in details['resistance_anomalies'].values() if v > 0.3)}个特征异常")
                else:
                    print(f" 系统正常")

        print(f"\n 基于风速计算风阻的矿井通风故障诊断系统完成!")
        print(f" 故障诊断准确率: {metrics['status_accuracy']:.4f}")
        print(f" 巷道定位准确率: {metrics.get('tunnel_accuracy', 0):.4f}")

        # 显示保存的文件
        print(f"\n 生成的文件:")
        project_dir = "D:/Project_python"
        if os.path.exists(project_dir):
            files = os.listdir(project_dir)
            for file in files:
                if file.endswith(('.keras', '.pkl', '.png', '.xls')):
                    print(f"  - {file}")

    except Exception as e:
        print(f" 系统运行失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()