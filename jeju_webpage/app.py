from flask import Flask, render_template, request, redirect, url_for
from clustering import predict_user_cluster  # clustering.py에서 클러스터링 함수 가져오기

app = Flask(__name__)

# 사용자 데이터 저장용 딕셔너리
user_data = {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/user_info', methods=['GET', 'POST'])
def user_info():
    if request.method == 'POST':
        # HTML 폼 데이터 받기
        user_data['nickname'] = request.form.get('nickname')
        user_data['gender'] = request.form.get('gender')
        user_data['age'] = request.form.get('age')
        user_data['companion'] = request.form.get('companion')
        user_data['stay_duration'] = int(request.form.get('stay_duration'))

        # 클러스터링 실행
        user_cluster = predict_user_cluster(
            user_data['nickname'],
            user_data['gender'],
            user_data['age'],
            user_data['companion'],
            user_data['stay_duration']
        )

        # 클러스터 결과 저장
        user_data['cluster'] = user_cluster

        # 다음 단계로 이동
        return redirect(url_for('select_keywords'))

    return render_template('user_info.html')

@app.route('/select_keywords')
def select_keywords():
    return render_template('select_keywords.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# cd "C:\Users\user\PycharmProjects\2025\project_JEJU\jeju_webpage"