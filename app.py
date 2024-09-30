from flask import Flask, render_template, request, jsonify
import base64
from kmeans import KMeans
import json

app = Flask(__name__)

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

            K = int(request.form.get('K', 3))
            init_method = request.form.get('init_method', 'random')
            num_points = int(request.form.get('num_points', 500))
            random_state = int(request.form.get('random_state')) if request.form.get('random_state') else None


            if not kmeans_instance or kmeans_instance.K != K or kmeans_instance.init_method != init_method or kmeans_instance.random_state != random_state:
                kmeans_instance = KMeans(K=K, init_method=init_method, random_state=random_state)


            kmeans_instance.generate_dataset(num=num_points)

            if init_method != 'manual':

                plot_url = kmeans_instance.run_initialization()
                message = "The data set was generated and the cluster center was initialized successfully.。"
            else:

                plot_url = kmeans_instance.run_initialization()
                message = "The data set was generated and the cluster center was initialized successfully."

                coordinates = kmeans_instance.coordinates.tolist()
                centroids = kmeans_instance.centroids.tolist() if kmeans_instance.centroids is not None else []
                plot_url = kmeans_instance.reset_centroids()

        elif action == 'iterate_once':
            if kmeans_instance:
                if kmeans_instance.centroids is not None:
                    plot_url, converged = kmeans_instance.fit_one_step()
                    if converged:
                        message = "The clustering has converged."
                    else:
                        message = "One iteration has been executed."
                else:
                    message = "Please select manual cluster centers first."
            else:
                message = "Please generate the dataset and initialize the cluster centers first."

        elif action == 'iterate_until':
            if kmeans_instance:
                if kmeans_instance.centroids is not None:
                    plot_url, converged = kmeans_instance.fit_until_converge()
                    if converged:
                        message = "The clustering has converged."
                    else:
                        message = "The maximum number of iterations has been reached."
                else:
                    message = "Please select manual cluster centers."
            else:
                message = "Please select manual cluster centers first."

        elif action == 'reset':
            if kmeans_instance:
                plot_url = kmeans_instance.reset_centroids()
                message = "Cluster centers have been reset."
            else:
                message = "Please generate the dataset and initialize the cluster centers first."

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
    print(f"Received centroids: {centroids}")
    if not centroids or len(centroids) != kmeans_instance.K:
        return jsonify({'status': 'error', 'message': f'Please provide exactly {kmeans_instance.K} centroids.'}), 400
    try:
        kmeans_instance.set_manual_centroids(centroids)

        plot_url = kmeans_instance.run_initialization()
        message = "Manual center point setting is successful."
        print(f"Set centroids: {kmeans_instance.centroids}")
        return jsonify({'status': 'success', 'plot_url': plot_url, 'message': message}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False,port= 3000)
