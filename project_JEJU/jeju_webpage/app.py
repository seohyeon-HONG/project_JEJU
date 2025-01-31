from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# 사용자 정보를 저장할 변수
user_data = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/user_info', methods=['GET', 'POST'])
def user_info():
    if request.method == 'POST':
        user_data['name'] = request.form.get('name')
        user_data['age'] = request.form.get('age')
        user_data['gender'] = request.form.get('gender')
        return redirect(url_for('select_keywords'))
    return render_template('user_info.html')

@app.route('/select_keywords', methods=['GET', 'POST'])
def select_keywords():
    if request.method == 'POST':
        selected_keywords = request.form.getlist('keywords')
        user_data['keywords'] = selected_keywords
        return redirect(url_for('recommend'))
    keywords = ['힐링', '자연', '포토존', '산책', '사진', '즐거움', '예술', '감동']
    return render_template('select_keywords.html', keywords=keywords)

@app.route('/recommend')
def recommend():
    # 예시 추천 로직
    recommended_places = ['스누피가든', '카멜리아힐', '아쿠아플라넷 제주']
    return render_template('recommend.html', user_data=user_data, places=recommended_places)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)

# cd "C:\Users\user\PycharmProjects\2025\project_JEJU\jeju_webpage"