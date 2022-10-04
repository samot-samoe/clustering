import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
import visualiser as vi
import seaborn as sns

st.markdown('''<h1 style='text-align: center; color: black;'
            >Кластеризация. Часть 2</h1>''', 
            unsafe_allow_html=True)

st.write("""
Данный сримлит предназначен для ознакомления с самой распространённой задачей обучения без учителя, 
а именно - задачей кластеризации

\nДанные подготовили сотрудники ЛИА РАНХиГС.
""")

st.markdown('''<h2 style='text-align: left; color: black;'
            >Пайплайн лабораторной работы:</h2>''', unsafe_allow_html=True)
img_pipeline = Image.open('pipeline.png') #
st.image(img_pipeline, use_column_width='auto', caption='Общий пайплайн для приложения') 

# my_data = pd.read_csv('final_customer_clustering_drop.csv')
my_data = pd.read_csv('final_customer_clustering_encode.csv')
#-------------------------О проекте-------------------------
expander_bar = st.expander("Перед тем, как начать:")
expander_bar.markdown(
    """
\n Задача *кластеризации* - задача обучения без учителя, заключающаяся в разобщении данных на несколько групп, именуемых кластерами

\n**Используемые библиотеки:** [streamlit](https://docs.streamlit.io/library/get-started), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), 
[matplotlib](https://matplotlib.org/stable/api/index.html), [scikit-learn](https://scikit-learn.org/stable/#).
\n **Полезно почитать:**[Основные методы кластеризации данных](https://habr.com/ru/company/ods/blog/325654/)
[PCA-1](https://habr.com/ru/post/304214/),[PCA-2](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%B3%D0%BB%D0%B0%D0%B2%D0%BD%D1%8B%D1%85_%D0%BA%D0%BE%D0%BC%D0%BF%D0%BE%D0%BD%D0%B5%D0%BD%D1%82)
[UMAP]()
""")
#------------------


#-----------------Elbow Visualization---------------
st.subheader('Определение количества кластеров')

st.write(
""" Теперь давайте перейдем к нашей задаче, состоящей в том, чтобы наиболее оптимальным
способом обособить элементы датасета по разным кластерам.
Начинать задачу кластеризации стоит с определения *оптимального* количества кластеров.
Одним из способов такого определения явлется "метод локтя". Применяя данный метод 
оптимальное количество кластеров должно находится в месте "сгиба".
""")
elbow_option1 = st.number_input('Выберите максимальное количество кластеров',min_value =2,max_value=50)

visual = vi.Visualise()

button_elbow = st.button('Построить график')
if button_elbow:
    visual = vi.Visualise()
    st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
    st.pyplot(visual.elbow(elbow_option1+1,my_data))


#-----------------Silhuete Visualization---------------
st.write("""
Ещё один метод проверки оптимального количества кластеров - метрика **силуэт**. 
Оценка силуэта для набора выборочных точек данных используется для измерения того, насколько плотными и хорошо разделенными 
являются кластеры. Посмотрим, какое количество кластеров даст наилучший результат. В случае с метрикой силуэт,
 находящейся в диапазоне от -1 до 1, чем больше значение, тем более плотно и качественно разделены кластеры. 
 Проверим, пользуясь тем, количеством кластеров, которое мы получили при помощи метода "локтя".
 \n *Главная особенность коэффициента **силуэт** от всех прочих метрик оценки 
 качества кластеризации заключается в том, что нам не нужно знать истинные метки предполгаемых классов.*
 \n Значительнейшим образом на качество кластерищации могут повлиять инструменты понижения размерности.
 \n Но стоит также помнить, что для реальных данных плотность кластеров не является основополагающей метрикой, по этому не стоит гнаться в 
 попытке определить количество кластеров за значением этой метрики

 """)
form = st.form('form')
option1= form.selectbox('Выберите способ кластеризации',
  ('KMeans','AgglomerativeClustering')) #,'SpectralClustering'

if option1 == 'KMeans':
    p1='KMeans'
if option1 =='SpectralClustering':
    p1='SpectralClustering'
if option1 == 'AgglomerativeClustering':
    p1 = 'AgglomerativeClustering'
# st.write(p1)

option2= form.selectbox('Выберите способ снижения размерности',
('UMAP','TSNE','PCA'))

if option2 =='UMAP':
    p2='UMAP'
if option2 == 'TSNE':
    p2 ='TSNE'
if option2 == 'PCA':
    p2 ='PCA'
option3= form.number_input('Выберите количество клaстеров', min_value=2, max_value = 50)
if option3:
    p3=option3
fin_button = form.form_submit_button("Проверить")
# button_cluster = st.button('Построить')
if fin_button:
    # nerek 
    visual = vi.Visualise()
    st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
    # st.write(p1)
    st.pyplot(visual.visualizer(cluster = p1,reductor = p2,n_clusters = p3,data = my_data))
    # st.write(vi.C)

# #-----------------adjusted_rand_score Block---------------

# st.subheader('Проверим на реальных классах')

# st.write("""Теперь, чтобы проверить, насколько наша кластеризация соответствует реальным классам,воспользуемся
# метрикой *adjusted_rand_score*
# """)
# button_adjusted = st.button('Проверить')
# if button_adjusted:
#     visual = vi.Visualise()
#     st.write(visual.adj(y=np.asarray(y_),y_true = vi.C))

#-------------------laboratory Block--------------------
final_data = pd.read_csv('final_customer_clustering_drop.csv')

st.subheader('Лабораторная')

st.write("""Теперь давайте присутпим к анализу данных. Наша цель - выявить оптимальное количество кластеров, на которые можно разбить наш датасет,
а затем рассмотреть по различным признакам, какие значения с каким кластером соотносятся, и сделать выводы о выявленных группах.
""")
st.subheader('Первый шаг')
st.write(""" На основе полученной нами информации из блоков выше мы могли сделать вывод о том, какое количество кластеров является для нас наиболее 
оптимальным. Теперь давайте попробуем произвести кластеризацию на интересующее нас количество кластеров. Повторяемся, при выборе кластеризатора
и уменьшителя размерности не стоит слишком сильно надеяться на метрику силуэт, но и забывать о ней не стоит

""")

# up_form =  st.form("Up Form")
fin_op_2 = st.selectbox('Выберите кластеризатор',
('KMeans','AgglomerativeClustering')) #'SpectralClustering',
if fin_op_2 == 'KMeans':
    fin_cl = 'KMeans'

if fin_op_2 == 'SpectralClustering':
    fin_cl = 'SpectralClustering'

if fin_op_2 == 'AgglomerativeClustering':
    fin_cl = 'AgglomerativeClustering'

fin_op_1 = st.selectbox('Выберите уменьшитель размерности ',
  ('UMAP','TSNE','PCA'))
form =  st.form("Form")
if fin_op_1 == 'UMAP':
    form =  st.form("Form1")
    fin_red = 'UMAP'
    comp_input = form.number_input("(n_components)Выберите кoличество компонентов,до которых будет сжато пространство",min_value = 2,max_value =4)
    neigh_input = form.number_input("(n_neighbors)Выберите количество соседей для каждого элемента",min_value = 2, max_value = 100)
    dist_input = form.number_input("(min_dist)Выберите минимальное расстояние",min_value = 0.00,max_value= 0.99)
    cl_input = form.number_input("Выберите количество кластеров",min_value = 2,max_value= 15)
    fin_button = form.form_submit_button("Построить график")
    if fin_button:
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        st.pyplot(visual.final_vis(reductor =fin_red,cluster =fin_cl,n_clusters = cl_input,n_components = comp_input,n_neighbors=neigh_input,min_dist=dist_input,data = final_data))
if fin_op_1 =='TSNE':
    form =  st.form("Form2")
    fin_red = 'TSNE'
    comp_input = form.number_input("Выберите кoличество компонентов, до которых будет сжато пространство",min_value = 0,max_value =4)
    perp_input = form.number_input("Выберите количество ближайших соседей",min_value = 0.00, max_value = 50.00)
    cl_input = form.number_input("Выберите количество кластеров",min_value = 2,max_value= 15)
    fin_button = form.form_submit_button("Построить график")
    if fin_button:
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        st.pyplot(visual.final_vis(reductor =fin_red,cluster =fin_cl,n_clusters = cl_input,n_components = comp_input,perplexity=perp_input,data = final_data))
    
if fin_op_1=='PCA':
    form =  st.form("Form3")
    fin_red = 'PCA'
    comp_input = form.number_input("Выберите кoличество компонентов/ отсечение по совокупной дисперии",min_value = 0.00,max_value =4.00)
    cl_input = form.number_input("Выберите количество кластеров",min_value = 2,max_value= 15)
    fin_button = form.form_submit_button("Построить график")
    if fin_button:
        if comp_input>=1.00:
            comp_input = int(comp_input)
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        st.pyplot(visual.final_vis(reductor =fin_red,cluster =fin_cl,n_clusters = cl_input,n_components = comp_input,data = final_data))
    
    
        
# if fin_button:
#     form =  st.form("Form4")
#     visual = vi.Visualise()
    # form.set_option('deprecation.showPyplotGlobalUse', False) #hide warning

#------------------------Analytics-----------------------

st.subheader("Второй шаг")
st.write(""" Теперь, проведя кластеризацию, посмотрим на то, какие группы у нас получились, и какие зависимости мы можем обнаружить
""")
data = pd.read_csv('customer_clustering2.csv')
data['Кластеры'] = vi.C

distr = st.checkbox('Посмотрим распределение кластеров')
if distr:
    pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60","#A4ABB2"]
    pl = sns.countplot(x=data["Кластеры"], palette= pal)
    pl.set_title("Распределение кластеров")
    st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
    st.pyplot(plt.show())

st.subheader('Третий шаг')
st.write("""Если расрпеделение кластеров показалось вам адекватным, если все кластеры приблизительно одинаковы и нет особенно маленьких или черезмерно больших,
то можно переходить к слудющему шагу, а именно - смотреть зависимости кластеров от различных, интересующих нас значений. Нас прежде всего должно интересовать
то, сколько покупатель тратит денег
""")
spend = st.checkbox('Посмотрим, в каком кластере наибольшие траты')
if spend:
    plt.figure()
    pl=sns.swarmplot(x=data["Кластеры"], y=data["Покупки"], color= "#CBEDDD", alpha=0.5 )
    pl=sns.boxenplot(x=data["Кластеры"], y=data["Покупки"], palette=pal)
    st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
    st.pyplot(plt.show())
    st.write(""" По построенным графикам легко определить, покупательская способность какого кластера является для нас наиболее интересной
    
    """)

st.subheader('Четвертый шаг')
st.write(""" Теперь давайте попробуем выяснить, в каком кластере совершалось наибольшее количество покупок со скидкой
""")
deals = st.checkbox('Посмотрим где нибольшее количество покупок со скидкой')
if deals:
    plt.figure()
    pl=sns.boxenplot(y=data["Покупки_со_скидкой"],x=data["Кластеры"], palette= pal)
    pl.set_title("Количество покупок со скидкой")
    st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
    st.pyplot(plt.show())

    st.write("""С большой долей вероятности, если кластеризация проведена верно, то кластер, интересующий нас по наибольшему количеству
    покупок, будет по крайней мере, в числе тех, что в наименьшей степени подвержен зависимости от покупки со скидкой. Пусть это будет небольшой подсказкой
    """)
st.subheader('Пятый шаг')
st.write(""" Теперь, исходя из результатов предыдущего исследования, попробуем посмотреть, как расрпеделятся между кластерами все имеющися в датасете покупки клиентов
""")

purch = st.checkbox('Посмотрим на распределение покупок')
if purch:
    Places =["Покупки_через_сайт", "Покупки_из_каталога", "Покупки_в_магазине",  "Посещение_сайта_за_месяц"] 

    for i in Places:
        plt.figure()
        sns.jointplot(x=data[i],y = data["Покупки"],hue=data["Кластеры"])#, palette= pal)
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        st.pyplot(plt.show())

st.subheader('Шестой шаг')
st.write("""В последнем шаге нашего исследования нам необходимо, пользуясь данными о семейном положении, возрасте, образовании составить приблизительный 
образ представителя каждого из кластеров
""")
profile = st.checkbox('Посмотрим, как в кластерах распределены покупатели')
if profile:
    Personal = [ "Маленькие_дети","Подростки", "Возраст", "Всего_детей", "Размер_семьи", "С_кем_живет"] # "Уровень_образования" "Родитель",

    for i in Personal:
        plt.figure()
        sns.jointplot(x=data[i], y=data["Покупки"], hue =data["Кластеры"], kind="kde")#, palette=pal)
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        st.pyplot(plt.show())

st.subheader('Выводы')
st.write(""" Теперь у нас есть достаточное количество информации для того, чтобы произвести анализ сформированных кластеров.В некоторых 
вопросах может быть несколько ответов.

""")
with st.form('Ответьте на все вопросы, чтобы успешно завершить лабораторную работу'):
    st.markdown('**Вопрос 1:** Какое количество кластеров является оптимальным для данного датасета?')
    question_1_wrong_1 = st.checkbox('2-3', value=False, key='1')
    question_1_right_2 = st.checkbox('4-5', value=False, key='2')
    question_1_wrong_3 = st.checkbox('6-7', value=False, key='3')
    question_1_wrong_4 = st.checkbox('8-9', value=False, key='4')

    st.markdown('**Вопрос 2:** Опираясь на результаты второго и третьего шагов лабораторной, какие выводы можно сделать?')
    question_2_right_1 = st.checkbox('Кластер с превалирующим количеством трат часто является одним из самых малых по численности',value=False, key='5')
    question_2_wrong_2= st.checkbox('Распределение трат в кластерах прям коррелирует с количеством элементов в кластерах', value=False, key='6')
    question_2_wrong_3= st.checkbox('Кластер наибольшего объема имеет наименьшее число покупок', value=False, key='7')

    st.markdown('**Вопрос 3:** Опираясь на результаты третьего и четвертого шагов лабораторной работы, какие выводы можно сделать?')
    question_3_right_1 = st.checkbox('Кластер с наибольшим числом трат также будет содержать в себе наименьшее количество покупок со скидкой', value=False, key='8')
    question_3_right_2 = st.checkbox('Число покупок со скидкой частично коррелирует с наименьшим количеством трат в рамках кластеров', value=False, key='9')
    question_3_wrong_3 = st.checkbox('Очевидна прямая зависимость количества трат и количества покупок со скидкой', value=False, key='10')
    
    st.markdown("**Вопрос 4:** Выберите выводы, которые мы можем сделать из последнего шага:")
    question_4_right_1 = st.checkbox('Покупатели с наибольшей способностью к тратам не имеют или имеют только одного ребенка', value=False, key='11')
    question_4_wrong_2 = st.checkbox('Покупатели с наименьшей покупательской способностью старше 60 лет', value=False, key='12')
    question_4_right_3 = st.checkbox('Покупатели с наименьшей покупательской способностью имеют одного и более ребенка ')


    if st.form_submit_button('Закончить тест и посмотреть результаты'):
        if question_1_right_2 and question_2_right_1 and question_3_right_1 and question_3_right_2 and question_4_right_1 and question_4_right_3 :
            st.markdown('''<h3 style='text-align: left; color: green;'
            >Тест сдан! Лабораторная работа завершена.</h3>''', 
            unsafe_allow_html=True) 
        else:
            st.markdown('''<h3 style='text-align: left; color: red;'
            >Тест не сдан! Где-то была допущена ошибка.</h3>''', 
            unsafe_allow_html=True) 

