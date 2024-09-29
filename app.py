from flask import Flask, render_template, request, jsonify
import base64
from kmeans import KMeans
import json

app = Flask(__name__)

# 全局变量存储 KMeans 实例
kmeans_instance = None

die1 = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    global kmeans_instance
    plot_url = None
    message = ""
    coordinates = []
    centroids = []
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'generate':
            # 获取用户输入
            K = int(request.form.get('K', 3))
            init_method = request.form.get('init_method', 'random')
            num_points = int(request.form.get('num_points', 500))
            random_state = int(request.form.get('random_state')) if request.form.get('random_state') else None

            # 创建 KMeans 实例
            if not kmeans_instance or kmeans_instance.K != K or kmeans_instance.init_method != init_method or kmeans_instance.random_state != random_state:
                kmeans_instance = KMeans(K=K, init_method=init_method, random_state=random_state)

            # 生成数据集
            kmeans_instance.generate_dataset(num=num_points)

            if init_method != 'manual':
                # 初始化聚类中心并获取初始图像
                plot_url = kmeans_instance.run_initialization()
                message = "数据集生成并初始化聚类中心成功。"
            else:
                # 手动初始化，不初始化聚类中心
                plot_url = kmeans_instance.run_initialization()
                message = "数据集生成成功。请在图上点击选择初始中心点。"
                # 传递数据点和当前中心点给模板（此时 centroids 为 None）
                coordinates = kmeans_instance.coordinates.tolist()
                centroids = kmeans_instance.centroids.tolist() if kmeans_instance.centroids is not None else []
                plot_url = kmeans_instance.reset_centroids()

        elif action == 'iterate_once':
            if kmeans_instance:
                if kmeans_instance.centroids is not None:
                    plot_url, converged = kmeans_instance.fit_one_step()
                    if converged:
                        message = "聚类已收敛。"
                    else:
                        message = "已执行一次迭代。"
                else:
                    message = "请先选择手动聚类中心点。"
            else:
                message = "请先生成数据集并初始化聚类中心。"

        elif action == 'iterate_until':
            if kmeans_instance:
                if kmeans_instance.centroids is not None:
                    plot_url, converged = kmeans_instance.fit_until_converge()
                    if converged:
                        message = "聚类已收敛。"
                    else:
                        message = "达到最大迭代次数。"
                else:
                    message = "请先选择手动聚类中心点。"
            else:
                message = "请先生成数据集并初始化聚类中心。"

        elif action == 'reset':
            if kmeans_instance:
                plot_url = kmeans_instance.reset_centroids()
                message = "聚类中心已重置。"
            else:
                message = "请先生成数据集并初始化聚类中心。"

        return render_template('index.html',
                               plot_url=plot_url,
                               message=message,
                               init_method=kmeans_instance.init_method if kmeans_instance else 'random',
                               K=kmeans_instance.K if kmeans_instance else 3,
                               coordinates=coordinates,
                               centroids=centroids)

    return render_template('index.html', plot_url=plot_url, message=message)

@app.route('/a', methods=['GET', 'POST'])
def set_manual_centroids_route():
    global kmeans_instance
    if not kmeans_instance:
        return jsonify({'status': 'error', 'message': 'KMeans instance not found.'}), 400
    data = request.get_json()
    centroids = data.get('centroids')
    print(f"Received centroids: {centroids}")  # 调试信息
    if not centroids or len(centroids) != kmeans_instance.K:
        return jsonify({'status': 'error', 'message': f'Please provide exactly {kmeans_instance.K} centroids.'}), 400
    try:
        kmeans_instance.set_manual_centroids(centroids)
        # 初始化聚类中心并获取图像
        plot_url = kmeans_instance.run_initialization()
        message = "手动中心点设置成功。"
        print(f"Set centroids: {kmeans_instance.centroids}")  # 调试信息
        return jsonify({'status': 'success', 'plot_url': plot_url, 'message': message}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False,port= 3000)
