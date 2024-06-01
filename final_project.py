import pandas as pd
from openai import OpenAI
import os
import warnings
import streamlit as st
import re
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from autogluon.tabular import TabularPredictor
from pathlib import Path
import pickle



warnings.filterwarnings('ignore')
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

st.set_page_config(layout="wide")

    
df = pd.read_csv('Bitamin_HR/final_df.csv') #upload_and_fill_csv()
df = df.fillna(0)

if df is None:
    st.write("CSV 파일을 업로드하세요.")


st.title("비타민 자소서 분석기")
api=st.text_input('API-KEY를 입력하세요')
os.environ["OPENAI_API_KEY"] = api
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

available_classes = df['기수'].unique()
available_classes_withall = ['전체']+list(available_classes)
selected_class = st.selectbox('비교하고 싶은 기수를 선택하세요:', available_classes_withall)


user_text_1 = st.text_area("지원 동기를 포함하여 자신을 자유롭게 소개해 주세요.", height=150)

user_text_2 = st.text_area("통계 관련 이수 교과명과 학습 내용을 적어주세요.", height=150)

user_text_3 = st.text_area("기초통계 외에 R, Python, SPSS, SQL 등의 언어나 머신러닝, 딥러닝, 데이터마이닝, 선형대수학 등에 대한 이론 / 실습 관련하여 학습하신 내용에 대해 적어주세요.", height=150)

user_text_4 = st.text_area("데이터 분석과 인공지능을 통해 해결하고 싶은 문제가 있다면 무엇인지 구체적으로 기술해주세요. (어떤 데이터셋을 가지고 무엇을 분석해보고 싶다는 내용도 함께 명시해주세요) 되도록 이에 대한 답을 해주시되, 이 질문이 너무 어려우면 동아리에 들어와 하고 싶은 활동, 가장 기대하는 바 등에 대해 적어주세요.", height=150)

user_text_5 = st.text_area("대학 재학 중 기억에 남는 활동에 대한 소개와 함께 활동 중 발생했던 문제를 어떻게 해결하였는지 구체적으로 말씀해주세요.", height=150)


def plot_capabilities(df, personal_df, selected_class):
    df = df.loc[df['합불'] == 'O']
    
    soft_skills = ["의사소통능력", "문제해결능력", "대인관계능력", "자기개발능력", "열정및의지력"]
    hard_skills = ["수학지식", "통계학지식", "머신러닝지식", "딥러닝지식", "데이터처리능력",
                   '데이터분석능력', '데이터시각화능력', '알고리즘구현능력', '인공지능모델링능력']
    task_score = ['해결하고싶은과제점수']
    
    df_selected = df.groupby('기수')[soft_skills + hard_skills + task_score].mean().reset_index()
    
    if selected_class == '전체':
        avg_soft_scores = df[soft_skills].mean()
        avg_hard_scores = df[hard_skills].mean()
        avg_task_score = df[task_score].mean()
    else:
        df_selected = df.loc[df['기수'] == selected_class]
        avg_soft_scores = df_selected[soft_skills].mean()
        avg_hard_scores = df_selected[hard_skills].mean()
        avg_task_score = df_selected[task_score].mean()

    personal_soft_scores = personal_df[soft_skills].iloc[0]
    personal_hard_scores = personal_df[hard_skills].iloc[0]
    personal_task_score = personal_df[task_score].iloc[0]

    # Bar Chart for Soft Skills
    fig1 = go.Figure(data=[
        go.Bar(name=f'{selected_class}기수 합격자의 평균 역량', x=soft_skills, y=avg_soft_scores, marker_color='blue'),
        go.Bar(name='당신의 점수', x=soft_skills, y=personal_soft_scores, marker_color='red')
    ])
    fig1.update_layout(barmode='group', title='당신의 soft skill graph')

    # Bar Chart for Hard Skills
    hard_skills_categories = hard_skills
    fig2 = go.Figure(data=[
        go.Bar(name=f'{selected_class}기수 합격자의 평균 역량', x=hard_skills_categories, y=avg_hard_scores, marker_color='blue'),
        go.Bar(name='당신의 점수', x=hard_skills_categories, y=personal_hard_scores, marker_color='red')
    ])
    fig2.update_layout(barmode='group', title='당신의 hard skill graph')

    # Bar Chart for Task Score
    fig3 = go.Figure(data=[
        go.Bar(name=f'{selected_class}기수 합격자의 평균 역량', x=task_score, y=avg_task_score, marker_color='blue'),
        go.Bar(name='당신의 점수', x=task_score, y=personal_task_score, marker_color='red')
    ])
    fig3.update_layout(barmode='group', title='해결하고 싶은 과제의 점수')

    # Displaying with Streamlit
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)


def soft_skill_generate(user_text_1,user_text_5):
    text = user_text_1 + user_text_5
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system",
            "content":
            """
            당신은 채용담당자입니다. 자기소개서 text가 들어오면 텍스트 전체를 분석하여 언급된 역량을 식별하고,
            아래 정의 된 soft skill을 참고하여 각 역량이 감지된 이유에 대해 설명해주세요.

            <Soft skill 정의>
            의사소통능력 : 상대방과 대화를 나누거나 문서를 통해 의견을 교환할 때, 상호 간에 전달하고자 하는 의미를 정확하게 전달할 수 있는 능력을 의미한다. 또한 글로벌 시대에서 필요한 외국어 문서이해 및 의사표현능력도 포함한다.
            문제해결능력 : 업무 수행 중 발생하는 여러 문제를 창조적, 논리적, 비판적 사고를 통해 올바르게 인식하고 적절히 해결하는 능력이다.
            대인관계능력 : 생활에서 협조적인 관계를 유지하고 조직구성원들에게 도움을 줄 수 있으며, 조직 내부 및 외부의 갈등을 원만히 해결하고, 상대방의 요구를 파악 및 충족시켜줄 수 있는 능력을 의미한다.
            자기개발능력 : 자신의 능력과 적성, 특성 등을 이해하고 목표 성취를 위해 스스로를 관리하며 개발해 나가는 능력을 의미한다.
            열정및의지력 : 활동에 열심히 참여하고자 하는 의지와, 포기하지 않고 목표를 달성하려고 노력하려는 사고방식을 의미한다.

            출력 형식은 다음과 같습니다.

            <감지된 역량>
            ##(#%) : ##
            ##(#%) : ## (감지된 역량은 여러개가 될 수 있음)

            """
            },
            {
            "role": "user",
            "content": """
            '저는 끊임없이 정진하고 싶은 사람입니다. 1학년 2학기 때 진행한 첫 프로젝트는 저의 졸업 후 진로를 엿볼 수 있는 뜻깊은 기회였으나 진행 과정에서 스스로 데이터사이언티스트라는 직업에 대한 지식이 매우 부족함을 느꼈습니다. 특히 복잡한 모델을 파이썬으로 구현하는 과정에서 기술적으로도 배울 것이 많다는 것을 느꼈습니다. 하지만 참여 과정에서, 단순히 전공강의를 열심히 듣는 것보다 직접 프로젝트를 능동적으로 하나 완성해보는 것이 제 전공에 대한 이해도를 확연히 높여줄 수 있다는 것을 절실히 깨달았습니다. 그 후 저는 졸업 전까지 다양한 프로젝트를 통해 제가 몸담고자 하는 직업을 경험해 봐야겠다는 다짐을 하였습니다. 비타민은 저와 비슷한 목표를 가진 대학생들과 팀을 이뤄 프로젝트를 할 수 있는 기회와 더불어, 제가 직면한 저의 전공 및 진로에 대한 고민을 공유할 수 있는 창구를 제공해준다는 점에서 현재의 저에게 가장 필요한 활동이라는 생각이 듭니다. 또한 스터디와 현직자의 강연 등을 통해 졸업 후 진로에 대한 고민 해결의 실마리를 찾아 나가며 ‘정진’하고 싶습니다.'
            위와 같 텍스트에서 역량을 추출해주세요.
            역량 추출 시 다음과 같은 조건을 따라야 합니다.
            <조건>
            1) 역량 정의에 존재하는 역량만 추출할 것.
            2) 이 때 드러난 역량이 여러 개이면 전부 표시할 것.
            3) 역량 별로, 탐지된 정도에 따라서 '역량 수치' 를 0%부터 100% 사이의 값으로 표현할 것.
            4) 해당 역량이 탐지되지 않을 경우 '역량 수치'는 0%, 아주 뚜렷하고 명백하게 탐지될 경우 '역량 수치'는 100%로 정의.
            5) 해당 역량이 탐지된 정도가 30% 미만이라면 추출하지 않을 것.
            6) 각 역량 별로 0%에서 100% 사이의 값이 도출되는 것이므로, 모든 역량의 총합 수치가 100%가 아니어야함.
            7) 역량은 %가 높은 순으로 출력.
            8) 역량은 세부적인 것을 나누지 않고 큰 정의만 출력할 것.
            9) 정의 된 Soft skill말고 다른 역량은 추출하지 않을 것
            """
            },
            {
            "role":'assistant',
            "content":
                """
                <감지된 역량>
                자기개발능력 (90%) : 전공의 이해도 증진을 위해 부족한 부분을 깨닫고, 프로젝트 참여라는 새로운 목표를 세우고 이를 성취하려 노력함.
                열정및의지력 (70%) : 본인의 부족함을 인지한 뒤에도, 프로젝트 및 활동에 정진하며 참여하고자 하는 의지를 보임.
                """
            },
            {
            "role": "user",
            "content": text + '''라는 텍스트에서 역량을 추출해주세요. 역량 추출 시 다음과 같은 조건을 따라야 합니다.
            <조건>
            1) 역량 정의에 존재하는 역량만 추출할 것.
            2) 이 때 드러난 역량이 여러 개이면 전부 표시할 것.
            3) 역량 별로, 탐지된 정도에 따라서 '역량 수치' 를 0%부터 100% 사이의 값으로 표현할 것.
            4) 해당 역량이 탐지되지 않을 경우 '역량 수치'는 0%, 아주 뚜렷하고 명백하게 탐지될 경우 '역량 수치'는 100%로 정의.
            5) 해당 역량이 탐지된 정도가 50% 미만이라면 추출하지 않을 것.
            6) 각 역량 별로 0%에서 100% 사이의 값이 도출되는 것이므로, 모든 역량의 총합 수치가 100%가 아니어야함.
            7) 역량은 %가 높은 순으로 출력.
            8) 역량은 세부적인 것을 나누지 않고 큰 정의만 출력할 것.
            9) 정의 된 Soft skill말고 다른 역량은 추출하지 않을 것
            '''
            }
        ],
        temperature=0,
        top_p=1
        )
    soft_skill = response.choices[0].message.content
    columns = ["의사소통능력", "문제해결능력", "대인관계능력", "자기개발능력", "열정및의지력"]

    # Initialize dictionary with zeros
    data = {col: 0 for col in columns}

    # Extract percentages
    matches = re.findall(r'(\w+능력) \((\d+)%\)', soft_skill)

    # Update dictionary with extracted values
    for match in matches:
        skill, value = match
        if skill in data:
            data[skill] = int(value)

    # Create dataframe
    soft_df = pd.DataFrame([data])
    return soft_df

def hard_skill_generate(user_text_2,user_text_3):
    text = user_text_2 + user_text_3
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system",
            "content":
            """
            당신은 채용담당자입니다. 자기소개서 text가 들어오면 텍스트 전체를 분석하여 언급된 역량을 식별하고,
            아래 정의 된 hard skill을 참고하여 각 역량이 감지된 이유에 대해 설명해주세요.

            <Hard skill 정의>
            수학지식 : 수학지식은 미적분, 선형대수학, 해석학, 확률 이론 등을 이해하고, 데이터 분석과 모델링에서 수학적 기법을 적용하여 문제를 해결하는 능력입니다.
            통계학지식 : 통계학지식은 가설 검정, 회귀 분석, 분산 분석, 확률 분포 등의 통계적 방법을 사용하여 데이터로부터 유의미한 통계적 결론을 도출하는 능력입니다.
            머신러닝지식 : 머신러닝지식은 지도학습, 비지도학습, 강화학습, 서포트 벡터 머신, 의사결정 나무, 랜덤 포레스트 등 다양한 머신러닝 알고리즘을 이해하고 적용하는 능력입니다.
            딥러닝지식 : 딥러닝지식은 CNN, RNN, LSTM, GAN 등 다양한 딥러닝 모델을 이해하고, 텐서플로우나 파이토치 같은 프레임워크를 사용하여 모델을 구현하고 최적화하는 능력입니다.
            데이터처리능력 : 데이터처리능력은 SQL, NoSQL 데이터베이스를 사용하여 데이터를 처리하고, ETL(Extract, Transform, Load) 프로세스를 통해 데이터를 준비하는 능력입니다.
            데이터분석능력 : 데이터분석능력은 파이썬의 판다스, 넘파이 라이브러리를 사용하여 데이터 분석을 수행하고, 의미 있는 인사이트를 도출하는 능력입니다.
            데이터시각화능력 : 데이터시각화능력은 매트플롯립, 시본, 태블로 등을 사용하여 데이터 시각화를 통해 복잡한 데이터를 직관적으로 전달하는 능력입니다.
            알고리즘구현능력 : 알고리즘구현능력은 정렬, 검색, 그래프 탐색 알고리즘 등을 설계하고, 파이썬, R, C 등의 프로그래밍 언어로 구현하는 능력입니다.
            인공지능모델링능력 : 인공지능모델링능력은 학습 데이터를 확보, 가공, 특징 추출, 품질 검증, 학습을 통해 최적화된 모델을 도출하고 활용하는 능력입니다. 이를 통해 기획된 인공지능 서비스의 목적을 달성하기 위해 최적화된 AI 모델을 설계하고 구현할 수 있습니다.
            출력 형식은 다음과 같습니다.

            <감지된 역량>
            ##(#%) : ##
            ##(#%) : ## (감지된 역량은 여러개가 될 수 있음)

            """
            },
            {
            "role": "user",
            "content": """
            '통계를 위한 수학 수업을 통해 선형대수학의 행렬 부분을, 회귀분석 수업을 통해 선형회귀와 다중회귀 이론을, SPSS와 R을 활용한 데이터분석 수업을 들었으며, 
            혼공 데이터분석파이썬 교재로 데이터 분석에 대한 기초를 공부하였고, 
            점프 투 파이썬과 점프 투 판다스 교재와 교재의 유튜브 강의로 파이썬과 판다스에 대해 공부하였습니다.'
            위와 같 텍스트에서 역량을 추출해주세요.
            역량 추출 시 다음과 같은 조건을 따라야 합니다.
            <조건>
            1) 역량 정의에 존재하는 역량만 추출할 것.
            2) 이 때 드러난 역량이 여러 개이면 전부 표시할 것.
            3) 역량 별로, 탐지된 정도에 따라서 '역량 수치' 를 0%부터 100% 사이의 값으로 표현할 것.
            4) 해당 역량이 탐지되지 않을 경우 '역량 수치'는 0%, 아주 뚜렷하고 명백하게 탐지될 경우 '역량 수치'는 100%로 정의.
            5) 해당 역량이 탐지된 정도가 50% 미만이라면 추출하지 않을 것.
            6) 각 역량 별로 0%에서 100% 사이의 값이 도출되는 것이므로, 모든 역량의 총합 수치가 100%가 아니어야함.
            7) 역량은 %가 높은 순으로 출력.
            8) 역량은 세부적인 것을 나누지 않고 큰 정의만 출력할 것.
            9) 정의 된 Hard skill말고 다른 역량은 추출하지 않을 것
            """
            },
            {
            "role":'assistant',
            "content":
                """
                <감지된 역량>
                데이터분석능력 (90%) : SPSS와 R을 활용한 데이터 분석 수업을 들었으며, 혼공 데이터분석파이썬 교재로 데이터 분석에 대한 기초를 공부하였고, 점프 투 파이썬과 점프 투 판다스 교재와 교재의 유튜브 강의로 파이썬과 판다스에 대해 공부하였습니다.
                통계학지식 (70%) : 회귀분석 수업을 통해 선형회귀와 다중회귀 이론을 학습하였습니다.
                수학지식 (60%) : 통계를 위한 수학 수업을 통해 선형대수학의 행렬 부분을 학습하였습니다.
                """
            },
            {
            "role": "user",
            "content": text + '''라는 텍스트에서 역량을 추출해주세요. 역량 추출 시 다음과 같은 조건을 따라야 합니다.
            <조건>
            1) 역량 정의에 존재하는 역량만 추출할 것.
            2) 이 때 드러난 역량이 여러 개이면 전부 표시할 것.
            3) 역량 별로, 탐지된 정도에 따라서 '역량 수치' 를 0%부터 100% 사이의 값으로 표현할 것.
            4) 해당 역량이 탐지되지 않을 경우 '역량 수치'는 0%, 아주 뚜렷하고 명백하게 탐지될 경우 '역량 수치'는 100%로 정의.
            5) 해당 역량이 탐지된 정도가 50% 미만이라면 추출하지 않을 것.
            6) 각 역량 별로 0%에서 100% 사이의 값이 도출되는 것이므로, 모든 역량의 총합 수치가 100%가 아니어야함.
            7) 역량은 %가 높은 순으로 출력.
            8) 역량은 세부적인 것을 나누지 않고 큰 정의만 출력할 것.
            9) 정의 된 Hard skill말고 다른 역량은 추출하지 않을 것
            '''
            }
        ],
        temperature=0,
        top_p=1
        )
    hard_skill = response.choices[0].message.content
    columns = ["수학지식", "통계학지식", "머신러닝지식", "딥러닝지식", "데이터처리능력",'데이터분석능력','데이터시각화능력','알고리즘구현능력','인공지능모델링능력']

    # Initialize dictionary with zeros
    data = {col: 0 for col in columns}

    # Extract percentages
    matches = re.findall(r'(\w+지식|\w+능력) \((\d+)%\)', hard_skill)

    # Update dictionary with extracted values
    for match in matches:
        skill, value = match
        if skill in data:
            data[skill] = int(value)

    # Create dataframe
    hard_df = pd.DataFrame([data])
    return hard_df

def Q4_socre_generate(user_text_4):
    text = user_text_4
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system",
            "content":
            """
            당신은 채용담당자입니다. 현재, '데이터 분석을 통해 해결하고 싶은 과제 또는 기대활동' 에 대해서 채점을 해야 하는 상황입니다.
            자기소개서 text가 들어오면 텍스트 전체를 읽고, 다음과 같은 <기준>에 따라 점수를 매긴 뒤 이에 대한 근거를 설명해주세요.

            <기준>
            1. 목표 명확성(40점) : 데이터 분석을 통해 해결하고자 하는 문제상황과, 수행하고자 하는 목표가 뚜렷하게 드러나는가? 뚜렷하게 드러날수록 높은 점수를 준다.
            2. 데이터셋 구상(30점) : 데이터셋에 대한 충분한 고민이 보이거나, 혹은 창의적인 데이터셋 아이디어가 확인되는가? 보이거나, 확인될수록 높은 점수를 준다.
            3. 고도화 여부(20점) : 데이터 분석 설계가 얕거나 허술하지 않고, 명확한 연구 설계가 완성되었는가? 자세하고 명확한 설계일수록 높은 점수를 준다.
            4. 실현 가능성(10점) : 텍스트에서 제시된 연구 계획이 현실적으로 실현 가능한 일인가? 구현이 가능한 계획일수록 높은 점수를 준다. 실현가능하다는 의미는 내용이 얼마나 구체적인지와는 무관하다.

            위와 같이, 총점은 100점 만점 입니다.

            출력 형식은 '기준명(##점) : 이유' 순서이며, 예시는 다음과 같습니다. 이때, 총점은 4개 기준의 점수를 모두 더한 값을 출력합니다.

            <분석 결과>
            목표 명확성(##점) : ##
            데이터셋 구상(##점) : ##
            고도화 여부(##점) : ##
            실현 가능성(##점) : ##
            
            총점 : ##점
            """
            },
            {
            "role": "user",
            "content": text + '''라는 텍스트를 읽고, 다음과 같은 <조건>에 따라 점수를 매긴 뒤 이에 대한 근거를 설명해주세요.
            
            <조건>
            1) 출력 결과는 <기준>의 순서를 따를 것.
            2) 지원자는 빅데이터 분석 동아리에 지원한 상황이며, 해당 질문은 데이터 분석 연구 설계 능력을 확인하는 데에 있음을 명심할 것.
            3) 내용이 과하게 간단하거나, 데이터 분석 연구 설계와 관련이 없는 내용으로 판단된다면 해당 기준에 대해 '0점'으로 처리할 것.
            '''
                
            }
        ],
        temperature=0,
        top_p=1
)
    Q4_skill = response.choices[0].message.content
    total_score = int(re.search(r'총점\s*:\s*(\d+)', Q4_skill).group(1))
    total_df=pd.DataFrame({'해결하고싶은과제점수':[total_score]})
    return total_df

def recommend_improvements(candidate_data, importance_df, top_n=3):
    candidate_with_importance = candidate_data.merge(importance_df, on='Feature')
    candidate_with_importance.sort_values(by=['Importance', 'Score'], ascending=[False, True], inplace=True)
    recommended_improvements = candidate_with_importance.head(top_n)
    return recommended_improvements[['Feature', 'Score']]

def recommend_based_on_average(candidate_scores, average_scores, top_n=3):
    improvement_needed = average_scores - candidate_scores
    improvement_needed = improvement_needed[improvement_needed > 0]
    recommendations = improvement_needed.sort_values(ascending=False).head(top_n)
    return recommendations

def load_model_and_recommend(candidate_id,your_df):
    # 모델과 피처 중요도 데이터 불러오기
    model_path = 'model'
    predictor = TabularPredictor.load('/Users/seoeunseo/Desktop/BITAmin/NLP/AutogluonModels/ag-20240601_165334')
    importance_df = pd.read_csv('/Users/seoeunseo/Desktop/BITAmin/NLP/AutogluonModels/ag-20240601_165334/feature_importances.csv')

    # 데이터 불러오기 및 id 열 붙이기
    file_path = '/content/drive/MyDrive/Colab Notebooks/Taeho/Bitamin_HR/'
    data = pd.read_csv('/Users/seoeunseo/Desktop/BITAmin/NLP/AutogluonModels/preprocessed_df.csv')
    data = pd.concat([data, your_df])

    # 지정된 id를 가진 후보자 데이터 선택
    candidate_data = data[data['id'] == candidate_id].drop(columns=['id'])

    if candidate_data.empty:
        return "Candidate ID not found in the dataset."

    # 모델을 사용한 합격 확률 예측
    predictions = predictor.predict_proba(candidate_data)
    candidate_data['pass_probability'] = predictions.iloc[:, 1]  # 'result' 레이블이 1일 확률

    # 피처 중요도 추출 및 지원자 역량 개선 권장
    recommendations = recommend_improvements(pd.DataFrame({
        'Feature': candidate_data.columns.drop('result'),
        'Score': candidate_data.iloc[0, :-1].values  # 'result'와 'pass_probability'를 제외한 모든 점수
    }), importance_df)

    print("합격 확률:", candidate_data['pass_probability'].iloc[0])
    print("\n피처별 중요도 기반 추천 역량:")
    print(recommendations)

    st.write("합격 확률:", candidate_data['pass_probability'].iloc[0])
    st.write("\n피처별 중요도 기반 추천 역량:")
    st.write(recommendations)

    # 합격자 데이터 선택 및 평균 점수 계산
    passers = data[data['result'] == 1]
    average_scores = passers.drop(['result', 'id'], axis=1).mean()

    # 합격자 평균과 지원자 점수 비교하여 추천
    recommendations_avg = recommend_based_on_average(candidate_data.drop(['result', 'pass_probability'], axis=1).iloc[0], average_scores)

    print("\n합격자 역량별 평균 점수 기반 추천 역량:")
    print(recommendations_avg)
    st.write("\n합격자 역량별 평균 점수 기반 추천 역량:")
    st.write(recommendations_avg)


# # 유저 ID를 입력으로 받음
# candidate_id_input = 12  # 예시 ID
# load_model_and_recommend(candidate_id_input)


if st.button("Analyze"):
    if user_text_1 and user_text_2 and user_text_3 and user_text_4 and user_text_5:
        with st.spinner("Analyzing..."):
            soft_df = soft_skill_generate(user_text_1, user_text_5)
            hard_df = hard_skill_generate(user_text_2, user_text_3)
            Q4_df = Q4_socre_generate(user_text_4)
            your_df = pd.concat([soft_df, hard_df, Q4_df], axis=1)
            your_df.insert(0, 'id', len(df)+1)
            candidate_id_input = your_df.iloc[:,0:1]
            st.success("Analysis Complete!")
            st.write('당신의 skill')
            st.dataframe(your_df)
            plot_capabilities(df, your_df, selected_class)
            st.write('지원자가 더하면 좋을 추천 역량')
            load_model_and_recommend(len(df)+1,your_df)
    else:
        st.error("Please input text in all fields.")