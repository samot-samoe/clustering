# импортируем библиотеки

from email import header
import streamlit as st

from PIL import Image
import visualiser as vi

import pandas as pd
import numpy as np

import datetime
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import plotly.graph_objects as go
import plotly.express as px

# DF = pd.read_csv(r'final_customer_clustering_drop.csv')

st.markdown('''<h1 style='text-align: center; color: black;'
            >Кластеризация часть 1: знакомство с методом и инструментами</h1>''', 
            unsafe_allow_html=True)

st.write("""
Данный сримлит предназначен для ознакомления с самой распространённой задачей обучения без учителя, а именно - задачей кластеризации. Основной задачей кластеризации является первичная разметка датасета, которая поможет выявить внутреннюю структуру данных.
Если говорить простыми словами, кластеризация нужна, чтобы упорядочить данные по различным группам в зависимости от их особенностей.

Полезно почитать: **[1](http://www.machinelearning.ru/wiki/index.php?title=Кластеризация)**, **[2](https://habr.com/ru/company/ods/blog/325654/)**

Стримлит "Кластеризация" состоит из 2 частей
\n**Первая часть**: познакомимся с разными методами и способами кластеризации на примере 2 популярных учебных датасетов
\n**Вторая часть**: ознакомление с различными метриками и способами определения оптимального количества кластеров, а также прохождение лабораторной работы

\nДанные подготовили сотрудники ЛИА РАНХиГС.
""")
st.markdown('''<h2 style='text-align: left; color: black;'
            >Пайплайн лабораторной работы:</h2>''', unsafe_allow_html=True)
img_pipeline = Image.open('pipeline.png') #
st.image(img_pipeline, use_column_width='auto', caption='Общий пайплайн для приложения') #width=450

# pipeline = Image.open('images/Pipeline_2.png')
# st.image(pipeline)

#-------------------------О проекте-------------------------

pipeline_description = st.expander("Описание пайплайна стримлита:")
pipeline_description.markdown(
    """
\nЗелёным обозначены этапы, корректировка которых доступна студенту, красным - этапы, которые предобработаны и скорректированы сотрудником лаборатории.
\n**1. Сбор данных:** был использован датасет из соревнований на платформе kaggle ([ссылка]())
\n**2. Первичная обработка данных:** избавление от пропущенных значений, создание новых столбцов, проверка и удаление выбросов
\n**3. Графический анализ данных** студенту предоставляется возможность ознакомиться с графиками распределния данных
\n**4. Настройка параметров:**составители лабораторной работы провели настройку модели
\n**5. Обучение модели:** студенту представляется возможность использовать настроенную модель для кластеризации
\n**6. Настройка параметров:** также студенту предоставляется возможность самому корректировать некоторые параметры моделей
\n**7. Веб-приложения Streamlit:** Кластеризация
\n**Используемые библиотеки:** [streamlit](https://docs.streamlit.io/library/get-started), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [sklearn](https://matplotlib.org/stable/api/index.html), 
[numpy](https://numpy.org/doc/stable/), [pillow](https://pillow.readthedocs.io/en/stable/), [matplorlib](https://matplotlib.org), [umap](https://umap-learn.readthedocs.io/en/latest/).
""")


expander_bar = st.expander("Перед тем, как начать:")
expander_bar.markdown(
    """
\n *Кластеризации* - относится к задаче обучения без учителя (когда у нас заранее нет ответов, по которым мы учим модель), которая заключается в разбиение данных на несколько групп (заранее нам количество неизвестно), 
именуемых кластерами.

\n**Используемые библиотеки:** [streamlit](https://docs.streamlit.io/library/get-started), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), 
[matplotlib](https://matplotlib.org/stable/api/index.html), [scikit-learn](https://scikit-learn.org/stable/#).
\n **Полезно почитать:**[Основные методы кластеризации данных](https://habr.com/ru/company/ods/blog/325654/)
[PCA-1](https://habr.com/ru/post/304214/),[PCA-2](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%B3%D0%BB%D0%B0%D0%B2%D0%BD%D1%8B%D1%85_%D0%BA%D0%BE%D0%BC%D0%BF%D0%BE%D0%BD%D0%B5%D0%BD%D1%82),
[PCA-3](https://wiki.loginom.ru/articles/principal-component-analysis.html)
[t-SNE-1](https://ru.wikipedia.org/wiki/%D0%A1%D1%82%D0%BE%D1%85%D0%B0%D1%81%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B5_%D0%B2%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5_%D1%81%D0%BE%D1%81%D0%B5%D0%B4%D0%B5%D0%B9_%D1%81_t-%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5%D0%BC)
,[t-SNE-2](https://habr.com/ru/post/267041/),
[UMAP-1](https://ru.wikipedia.org/wiki/UMAP)
[UMAP-2](https://habr.com/ru/company/newprolab/blog/350584/)

""")

df_info_expander = st.expander("Информация о датасете:")
df_info_expander.markdown(
"""
\n**customers_clustering.csv** - набор данных, о покупках в магазине за 2 года (2012-2014 года), который включает в себя данные о покупателях, количество приобретенных продуктов по категориям и другие параметры.
\n**[Ссылка на данные](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis?datasetId=1546318&sortBy=voteCount)**
""")


col_expander = st.expander('Описание столбцов:')
col_expander.markdown("""
\n**Год_рождения** - Год рождения клиента
\n**Уровень образования** - Уровень образования клиента 
\n**Семейное_положение** - Семейное положение клиента
\n**Доход** - Годовой уровень дохода клиента 
\n**Маленькие_дети** - Количество маленьких детей у клиента 
\n**Подростки** - Количество подростков у клиента 
\n**Регистрация** - Первая запись о клиенте в компании
\n**Последняя_закупка** - Количество дней с последней покупки клиента
\n**Вино** - Сумма, потраченная на вино за последние 2 года
\n**Фрукты** - Сумма, потраченная на фрукты за последние 2 года
\n**Мясо** - Сумма, потраченная на мясо за последние 2 года
\n**Рыба** - Сумма, потраченная на рыбу за последние 2 года
\n**Сладкое** - Сумма, потраченная на сладкое за последние 2 года
\n**Золото** - Сумма, потраченная на золото за последние 2 года
\n**Покупки_со_скидкой** - Количество покупок, совершенных по скидке
\n**Покупки_через_сайт** - Количество покупок, соверщенных через веб-сайт компании
\n**Покупки_из_каталога** - Количество покупок, сделанных с использованием каталога
\n**Покупки_в_магазине** - Количество покупок, сделанных непосредственно в магазинах
\n**Посещение_сайта_за_месяц** - Количество посещений веб-сайта компании за последний месяц
\n**Реклама_3** - Сделал ли клиент покупку после 3 рекламной компании (1 - да, 0 - нет)
\n**Реклама_4** - Сделал ли клиент покупку после 4 рекламной компании (1 - да, 0 - нет)
\n**Реклама_5** - Сделал ли клиент покупку после 5 рекламной компании (1 - да, 0 - нет)
\n**Реклама_1** - Сделал ли клиент покупку после 1 рекламной компании (1 - да, 0 - нет)
\n**Реклама_2** - Сделал ли клиент покупку после 2 рекламной компании (1 - да, 0 - нет)
\n**Жалобы** - Были ли жалобы у клиента за последние 2 года (1 - да, 0 - нет)
\n**Последняя_рекламная_компания** - Сделал ли клиент покупку после последней рекламной компании (1 - да, 0 - нет)
""")




df = pd.read_csv('customer_clustering.csv')

st.subheader('Анализ данных')

show_df = st.checkbox('Показать Датасет')
if show_df:
  st.dataframe(df)
  # number = st.number_input('Сколько строк показать',min_value =1, max_value=df.shape[1])
  # st.dataframe(df.head(number))

data_shape = st.checkbox('Размер Датасета')
if data_shape:
    shape = st.radio(
    "Выбор данных",
    ('Строки', 'Столбцы'))
    if shape == 'Строки':
        st.write('Количество строк:', df.shape[0])
    elif shape == 'Столбцы':
        st.write('Количество столбцов:', df.shape[1])


values = st.checkbox('Уникальные значения переменной')
if values:
  cols = st.multiselect('Выбрать столбец', 
  df.columns.tolist())
  if cols:
    st.write(pd.DataFrame(df[cols].value_counts(), columns=['количество уникальных значений']))

data_types = st.checkbox('Типы данных')
if data_types:
  st.write('**Тип данных** - внутреннее представление, которое язык программирования использует для понимания того, как данные хранить и как ими оперировать')
  type_info = st.expander('Информация об основных типах данных')
  type_info.info('''Object - текстовые или смешанные числовые и нечисловые значения 
  \nINT - целые числа 
  \nFLOAT - дробные числа 
  \nBOOL - значения True/False
  \nDATETIME - значения даты и времени
  ''')
  st.write(pd.DataFrame(df.dtypes.astype('str'), columns=['тип данных']))

data_describe = st.checkbox('Описательная статистика по всем числовым столбцам')
if data_describe:
  describe_expander_ = st.expander('Информация о данных, которые входят в описательную статистику')
  describe_expander_.info('''Count - сколько всего было записей 
  \nMean - средняя велечина 
  \nStd - стандартное отклонение
  \nMin - минимальное значение
  \n25%/50%/70% - перцентили (показывают значение, ниже которого падает определенный процент наблюдений. Например, если число 5 - это 25% перцентиль, значит в наших данных 25% значений ниже 5)
  \nMax - максимальное значение
  ''')
  st.dataframe(df.describe())

non_val = st.checkbox('Пропущенные значения')
if non_val:
  st.write(pd.DataFrame(df.isnull().sum().sort_values(ascending=False), columns=['количество пропущенных значений']))

#-----------------Preprocessing ---------------

st.subheader('Предварительная обработка данных')

st.write('Давайте сначала обработаем наши данные')

# with st.expander('Первый шаг'):
#with st.form(key='customer_for'):
if st.checkbox('Первый шаг'):
  st.markdown('''
  Для начала изабавимся от пропущенных значений.
  ''')
  if st.checkbox('Удалить пропуски'):
    df = df.dropna()
    st.write("Сколько осталось строк:", len(df))

if st.checkbox('Второй шаг'):
  st.markdown('''Приведем столбец "Регистрация" к типу данных datetime и посмотрим, когда была первая запись, а когда - последняя. Затем создадим новый столбец "Дней_в_магазине". Чтобы получить этот 
  столбец, мы вычтем последнюю дату "регистрации" в данных от даты регистрации у каждого клиента. Так мы получим информацию, сколько дней человек является нашим клиентом''')
  if st.checkbox('Поменять тип данных и получить информацию'):
    df['Регистрация'] = pd.to_datetime(df['Регистрация'])
    dates = []
    for i in df["Регистрация"]:
      i = i.date()
      dates.append(i)  

    days = []
    max_day = max(dates)
    for i in dates:
      delta = max_day - i
      days.append(delta.days)
    df["Дней_в_магазине"] = days

    st.write("Самая первая запись клиента:", min(dates))
    st.write("Самая последняя запись клиента:", max(dates))
    st.dataframe(df['Дней_в_магазине'])
 
if st.checkbox('Третий шаг'):
  st.write('''Сделаем еще несколько преобразований, создадим новые производные столбцы
  \n Столбец "Возраст": вычтем 2014 из года рождения
  \n Столбец "Покупки": просуммируем покупки по всем категориям
  \n Столбец "С_кем_живет": в зависимости от семейного статуса будет указывать с партнером или один
  \n Столбец "Всего_детей": Посчитаем общее количество детей (дети + подростки)
  \n Столбец "Размер_семьи": просуммируем информацию по столбцам "все_дети" и "с_кем_живет"
  \n Столбец "Родитель": 1 - если есть дети, 0 - если нет
  \n Столбец "Образование" соберем в 3 категории: студент (еще учатся), выпускник, магистр/аспирант
  ''')
  if st.checkbox('Применить преобразования'):
    df['Возраст'] = 2014-df["Год_рождения"]

    df['Покупки'] = df['Мясо'] + df['Фрукты'] + df['Рыба'] + df['Вино'] + df['Золото'] + df['Сладкое']

    df['С_кем_живет'] = df['Семейное_положение'].replace({'Married': 'Партнер', 'Together': 'Партнер', 'Absurd': 'Один', 'Widow': 'Один', 'YOLO': 'Один', 'Divorced': 'Один', 'Single': 'Один', "Alone":'Один'})

    df['Всего_детей'] = df['Маленькие_дети'] + df['Подростки']

    df['Размер_семьи'] = df['С_кем_живет'].replace({'Один': 1, 'Партнер':2}) + df['Всего_детей']

    df['Родитель'] = np.where(df['Всего_детей'] > 0, 1, 0)

    # df["Уровень образования"]=df["Уровень образования"].replace({"Basic":"студент","2n Cycle":"студент", "Graduation":"студент", "Master":"магистр/аспирант", "PhD":"магистр/аспирант"})

    df = df.drop(['Год_рождения', "Семейное_положение", 'Регистрация'],axis=1)
    
    

    st.dataframe(df)
#------------------------------EDA------------------------------------


if st.checkbox('Четвертый шаг'):
  st.markdown("""
  Давайте снова взглянем на наши данные:
  """)
  st.write(df.describe())
  st.markdown('''
  Обратите внимание на столбцы "Возраст" и "Доходы", кажется, что у нас есть сильные выбросы в этих данных. Чтобы проверить, можно построить ящик с усами. Давайте это и сделаем
  ''')
  if st.checkbox('Проверить выборосы в столбце с помощью ящика с усами'):
    box_one = px.box(df, y='Возраст')
    box_two = px.box(df, y='Доход')
    st.plotly_chart(box_one)
    st.plotly_chart(box_two)
    st.markdown('''
    Мы оказались правы, есть серьезные выбросы в наших данных. Давайте удалим данные клиента с доходом 666.666 и трех клиентов, которым больше 110 лет"
    ''')
    if st.checkbox('Удалить данные'):
      df = df[(df["Возраст"]<90)]
      df = df[(df["Доход"]<600000)]
      st.write('Вот так теперь выглядят ящики с усами')
      st.plotly_chart(px.box(df, y='Возраст'))
      st.plotly_chart(px.box(df, y='Доход'))
      
      



#-----------------Reductor Visualization---------------

st.subheader('Интструменты снижения размерностей')
st.write("""
Теперь, прежде чем говорить о кластеризации, необходимо рассказать об алгоритмах понижения размерности. Перед тем, как кластеризировать объекты, нам необходимо перевести их в пространство меньшей размерности (2-ух или 3-ех мерной).
На самом деле понижение размернорности применяют и для других целей (визуализация данных, предобработка признаков перед обучением для борьбы с зашумленными данными), но нас сейчас это не интересует.

\n Использование алгоритмов кластеризации на датасете с большим количеством фичей редко приносит нужный результат, что демонстрируется как метриками (о которых 
мы поговорим чуть позже), так визуализацией данных. Поэтому для использования алгоритмов кластеризации предварительно используются алгоритмы понижения размерности.
\n Алгоритмы понижения размерности можно разделить на 2 основные группы: 
они пытаются сохранить либо глобальную структуру данных, либо локальные расстояния между точками. К первым относятся такие алгоритмы как PCA
(Метод главных компонент), ко вторым t-SNE и UMAP. 
\n Все визуализации работы алгоритмов уеньшения размерности производятся при помощи кластеризатора KMeans
\n *Отдельно стоит обратить внимание на то, что в данной работе приведены не все способы уменьшения размерности, но те, которые показались автору
наиболее часто используемыми*
""")
options_re = st.selectbox('Выберите инструмент ',
  ('PCA','t-SNE','UMAP'))

# -------------------------------------PCA----------------------------------------------------------
# dataframe = pd.read_csv('final_customer_clustering_drop.csv')
dataframe = pd.read_csv('final_customer_clustering_encode.csv')

if options_re == 'PCA':
    expander_bar = st.expander('Описание принципа работы PCA: ')
    expander_bar.markdown(
    """
    \n Алгоритм PCA (Метод главных компонент), в отличае от двух предыдущих, выстраивает оси новой системы координат пространства сниженной размерности
    (обычно их 2 или 3) таким образом, чтобы [дисперсия](https://ru.wikipedia.org/wiki/%D0%94%D0%B8%D1%81%D0%BF%D0%B5%D1%80%D1%81%D0%B8%D1%8F_%D1%81%D0%BB%D1%83%D1%87%D0%B0%D0%B9%D0%BD%D0%BE%D0%B9_%D0%B2%D0%B5%D0%BB%D0%B8%D1%87%D0%B8%D0%BD%D1%8B) 
    вдоль новых осей была максимальной из возможных
    \n [PCA](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%B3%D0%BB%D0%B0%D0%B2%D0%BD%D1%8B%D1%85_%D0%BA%D0%BE%D0%BC%D0%BF%D0%BE%D0%BD%D0%B5%D0%BD%D1%82)
    """)
    settings_bar = st.expander('Параметры алгоритма PCA: ')
    settings_bar.markdown(""" 
    \n Основным параметром алгоритма PCA является n_components. В отличае от алгоритмов UMAP и t_SNE здесь 
    n_components означает  число фичей, или количество измерений, которые вносят наибольшую значимость (обладают наибольшей дисперсией) в нашу модель.
    В среднем нам нужно от 95% до 99% дисперсии. Для вычисления оптимального набора компонентов воспользуемся функцией ниже.
    """)
    form = st.form("form")
    p = form.number_input("Выберите количество компонентов", min_value = 2, max_value = 28)
    button_ncomponents = form.form_submit_button("Выявить оптимальное количество компонентов")
    if button_ncomponents:
      visual = vi.Visualise()
      st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
      st.pyplot(visual.n_component(n_component = p,data = dataframe))

    form =  st.form("Form")
    p = form.number_input("Выберите кoличество компонентов",min_value = 2,max_value =28)
    h = form.number_input("Выберите количество кластеров",min_value = 2,max_value= 15)
    button_pca = form.form_submit_button("Построить график для уменьшителя размерности PCA")
    if button_pca:
        visual = vi.Visualise()
        # if p>=1.00:
            # p = int(p)
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        st.pyplot(visual.reductor_vis(reductor = 'PCA',n_clusters = h,n_components = p,data = dataframe))


#-----------------------------------TSNE-----------------------------------

if options_re == 't-SNE':
    expander_bar = st.expander('Описание принципа работы TSNE: ')
    expander_bar.markdown(
    """
    \n t_SNE работает следующим образом:
    \n 1. При помощи метода, фиксирующего похожесть элементов данных формируется распределение вероятности таким образом, что с большей вероятностью будут выбраны два похожих объекта.
    \n 2. Затем алгоритм формирует новое, похожее распределение вероятностей для низкоразмерного пространства.
    \n Если простыми словами: хотим сохранить расстояние между объектами, которые изначально находились близко и попытаться отдалить объекты, которые изначально находились далеко, 
    и построить это в новом уменьшенном пространстве.
    \n Этот алгоритм не используется для генерации новых признаков, скорее его используют для визуализации данных.
    """)
    expander_bar = st.expander('Описание основных параметров: ')
    expander_bar.markdown("""
    \n n_components - количество измерений, до которого снижается размерность датасета
    \n perplexity = Параметр perplexity связан с количеством ближайших соседей, которые используются в других алгоритмах уменьшения размерности. 
    Большие наборы данных обычно требуют большего значения этого параметра. Попробуйте выбрать значение от 5 до 50. Различные значения могут привести к 
    существенно разным результатам.
    
    """)
    form =  st.form("Form")
    comp_input = form.number_input("Выберите кoличество компонентов, до которых будет сжато пространство",min_value = 0,max_value =4)
    perp_input = form.number_input("Выберите количество ближайших соседей",min_value = 0.00, max_value = 50.00)

    h = form.number_input("Выберите количество кластеров",min_value = 2,max_value= 15)
    button_pca = form.form_submit_button("Построить график для уменьшителя размерности TSNE")
    if button_pca:
        visual = vi.Visualise()
        if p>=1.00:
            p = int(p)
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        st.pyplot(visual.reductor_vis(reductor = 'TSNE',n_clusters = h,n_components = comp_input,perplexity = perp_input,data = dataframe))
  
#--------------------------------------UMAP------------------------------
if options_re == 'UMAP':
    expander_bar = st.expander('Описание принципа работы UMAP: ')
    expander_bar.markdown(
    """
    \n Сама история возникновения алгоритма UMAP тесно связана с алгоритмом понижения размерности t-SNE. В основании обоих этих
    алгоритмов лежит два основных шага. Для UMAP они следующие:
    \n 1. При помощи метода, фиксирующего близость-дальность расположения элементов датасета формируется граф, 
    где каждый элемент соединен со своими ближаишими сосоедями (параметр количество ближайших соседей можно задавать)
    \n 2. Затем алгоритм создаёт новый граф в низкоразмерном пространстве, приближая его к старому.
    \n Количеством компонентов обозначается количество измерений, до которого будет сжат датасет.
    \n Подробнее об алгоритме UMAP: 
    [Статья в википедии](https://ru.wikipedia.org/wiki/UMAP), 
    [Документация для алгоритма UMAP](https://umap-learn.readthedocs.io/en/latest/basic_usage.html)
    """)
    expander_bar = st.expander('Описание основных параметров: ')
    expander_bar.markdown(
    """
    \n n_components - количество измерений, до которого снижается размерность датасета
    \n n_neighbours - Этот параметр управляет тем, как UMAP уравновешивает локальную и глобальную структуру данных. Это достигается за счет ограничения 
    размера локальной окрестности, на которую UMAP будет смотреть при попытке изучить многообразную структуру данных. Это означает, что низкие значения 
    n_neighbors заставят UMAP сосредоточиться на очень локальной структуре 
    (потенциально в ущерб общей картине), в то время как большие значения заставят UMAP рассматривать более крупные 
    окрестности каждой точки при оценке многообразной структуры данных. потеря мелкой детализированной структуры ради получения более широких данных.
    \n *по дефолту значение n_neighbours = 15*
    \n Параметр min_dist определяет, насколько плотно UMAP может упаковывать точки вместе. Это буквально означает минимальное расстояние друг от друга, на 
    котором точки могут находиться в низкоразмерном представлении. Это означает, что низкие значения min_dist приведут к более громоздким построениям. 
    Это может быть полезно, если вы заинтересованы в кластеризации или более тонкой топологической структуре. Большие значения min_dist предотвратят объединение 
    точек UMAP и вместо этого сосредоточатся на сохранении широкой топологической структуры.
    \n *по дефолту значение min_dist = 0.1, в диапазоне от 0.00 до 0.99*
    """)
    form =  st.form("Form")
    comp_input = form.number_input("(n_components)Выберите кoличество компонентов,до которых будет сжато пространство",min_value = 1,max_value =4)
    neigh_input = form.number_input("(n_neighbors)Выберите количество соседей для каждого элемента",min_value = 1, max_value = 100)
    dist_input = form.number_input("(min_dist)Выберите минимальное расстояние",min_value = 0.00,max_value= 0.99)
    cl_input = form.number_input("Выберите количество кластеров",min_value = 2,max_value= 15)

    button_pca = form.form_submit_button("Построить график для уменьшителя размерности UMAP")
    if button_pca:
        visual = vi.Visualise()
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        st.pyplot(visual.reductor_vis(reductor = 'UMAP',n_clusters = cl_input,n_components = comp_input,min_dist=dist_input,n_neighbors = neigh_input,data = dataframe))


#--------------------------------------Clustering-----------------------------------------------------

dataframe = pd.read_csv('final_customer_clustering_drop.csv')
st.subheader('Кластеризаторы')
st.write("""
Начать разговор о задачах кластеризации стоит собственно с алгоритмов кластеризации. Ниже представленны те алгоритмы, 
которые автору показались наиболее часто используемыми. Об их работе в сжатом виде можно прочесть ниже, более подробную информацию можно получить перейдя по ссылке
в конце описания работы кластеризатора
""")
options_cl = st.selectbox('Выберите кластеризатор',
  ('KMeans','AgglomerativeClustering')) #'SpectralClustering''DBSCAN'
if options_cl == 'KMeans':
    expander_bar = st.expander('Описание принципа работы KMeans: ')
    expander_bar.markdown(
    """
    \n Метод k-средних работает следующим способом:
    \n 1. Задаётся число k - кластеров и выбирается метрика проверки(о метриках см. ниже)
    \n 2. Выбирается k - случайных точек (центроид) из выборки
    \n 3. Находится по точке, ближайших к *k - случайных точек* 
    \n 4. Посредством усреднения координат точек определяются центры кластеров, 
    \n 5. После чего находится точка, ближайшая к новообразованному кластеру
    \n 6. Затем так повторяется до тех пор, пока вся выборка не окажется внутри кластера
    \n Подробнее об алгоритме KMeans:[Статья в википедии](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_k-%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D0%B8%D1%85), 
    [Документация для алгоритма KMeans ](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    """)
    option_kmeans_n= st.number_input('Выберите количество кластеров', min_value=2, max_value = 50)
    if option_kmeans_n:
        p=option_kmeans_n

    button_kmeans = st.button('Построить график Kmeans')
    if button_kmeans:
        visual = vi.Visualise()
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        st.pyplot(visual.proto_vis(cluster = 'KMeans',n_clusters = p,data = dataframe))

# if options_cl == 'SpectralClustering':
#     expander_bar = st.expander('Описание принципа работы Spectral clustering: ')
#     expander_bar.markdown(
#         """
#         \n Спектральная кластеризация - улучшеный вариант метода k-средних, в котором предварительно применяется
#         снижение размерности при помощи [матрицы сходства](https://ru.wikipedia.org/wiki/%D0%9C%D0%B0%D1%82%D1%80%D0%B8%D1%86%D0%B0_%D0%BC%D0%B5%D1%80_%D0%BA%D0%BE%D0%BD%D0%B2%D0%B5%D1%80%D0%B3%D0%B5%D0%BD%D1%86%D0%B8%D0%B8) 
#         \n *Стоит отдельно отметить, что Спектральная кластеризация особенно хорошо работает с небольшим количеством кластеров*
#         \n Подробнее об алгоритме SpectralCLustering:[Статья в википедии](https://ru.wikipedia.org/wiki/%D0%A1%D0%BF%D0%B5%D0%BA%D1%82%D1%80%D0%B0%D0%BB%D1%8C%D0%BD%D0%B0%D1%8F_%D0%BA%D0%BB%D0%B0%D1%81%D1%82%D0%B5%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F), 
#         [Документация для алгоритма Spectral Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)
#         """)
#     option_sc_n= st.number_input('Выберите количество кластеров', min_value=2, max_value = 50)
#     if option_sc_n:
#         p=option_sc_n
    
#     button_spectral = st.button('Построить график SpectralClustering ')
#     if button_spectral:
#         visual = vi.Visualise()
#         st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
#         st.pyplot(visual.proto_vis(cluster = 'SpectralClustering',n_clusters = p,data = dataframe))

if options_cl == 'AgglomerativeClustering':
    expander_bar = st.expander('Описание принципа работы Agglomerative Clustering: ')
    expander_bar.markdown(
        """
        \n Объект AgglomerativeClustering выполняет иерархическую кластеризацию по принципу «снизу вверх»: 
        каждое наблюдение начинается в своем собственном кластере, и кластеры последовательно объединяются.
        \n *Иерархическая кластеризация - это общее семейство алгоритмов кластеризации, которые создают вложенные кластеры путем 
        их последовательного слияния или разделения.*
        \n Подробнее об алгоритме Agglomerative Clustering:[Статья в википедии об иерархической кластеризации](https://ru.wikipedia.org/wiki/%D0%98%D0%B5%D1%80%D0%B0%D1%80%D1%85%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D0%BA%D0%BB%D0%B0%D1%81%D1%82%D0%B5%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F), 
        [Документация для алгоритма Agglomerative Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
        """)
    option_ag_n= st.number_input('Выберите количество кластеров', min_value=2, max_value = 30)
    if option_ag_n:
        p=option_ag_n
    
    button_ag = st.button('Построить график Agglomerative Clustering ')
    if button_ag:
        visual = vi.Visualise()
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        st.pyplot(visual.proto_vis(cluster = 'AgglomerativeClustering',n_clusters = p,data = dataframe))

# if options_cl == 'DBSCAN':
#   expander_bar = st.expander('Описания принципа работы DBSCAN: ')
#   expander_bar.markdown(
#     """
#     \n Алгоритм DBSCAN рассматривает кластеры как области высокой плотности, разделенные областями низкой плотности. Из-за этого довольно общего 
#     представления кластеры, найденные с помощью DBSCAN, могут иметь любую форму, в отличие от K_means, которые предполагают, что кластеры имеют 
#     выпуклую форму.
#     \n *Важное замечание, что алгоритм DBSCAN не предполагает указывания изначального количества кластеров*
#     \n Подробнее об алгоритме DBSCAN:
#     [Хорошее видео для понимания принципа работы алгоритма](https://www.youtube.com/watch?v=RDZUdRSDOok),
#     [Статья в википедии](https://ru.wikipedia.org/wiki/DBSCAN),
#     [Документация для алгоритма DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
#     """)
#   # option_dbs_n = st.number_input('Выберите количество кластеров', min_value=2, max_value = 30)
#   # if option_dbs_n:
#   #   p = option_dbs_n
#   button_dbs = st.button('Построить график DBSCAN')
#   if button_dbs:
#     visual = vi.Visualise()
#     st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
#     st.pyplot(visual.proto_vis(cluster = 'DBSCAN', data = dataframe))



#-------------------------------------Test-------------------------------------------------------
# st.subheader('Тест')
# form = st.form('test_form')
# genre = form.radio("Какой алгоритм кластеризации предполагает наличие центроид?", 
#                 ((''), ('Agglomerative clustering'), ('KMeans'),('Spectral Clustering')), index=0)
# genre_answer = 'KMeans'
# # if genre == 'KMeans':
#   # form.success('Ответ верный!')
#   # st.balloons()
# # else:
#   # form.error('Ответ неверный!') 

# genre2 = form.radio("Чем уменьшитель размерности PCA отличается от двух других, упомянутых",
#                  ((''),('PCA в качестве главного инструмента использует построение графа, где каждый элемент связывается с n-ближайших соседей'),
#                  ('PCA пытается отобразить все кластеры в целом, меньше акцентируя внимание на внутреннюю структуру, в то время как T-sne и UMAP пытается сохранить ближайшее окружение каждого элемента'),
#                  ('PCA в отличае от T-sne и UMAP предназначен для работы лишь с небольшим количеством данных')))
# # if genre2 == 'PCA пытается отобразить все кластеры в целом, меньше акцентируя внимание на внутреннюю структуру, в то время как T-sne и UMAP пытается сохранить ближайшее окружение каждого элемента':
#   # st.success('Ответ верный!')
# genre2_answer = 'PCA пытается отобразить все кластеры в целом, меньше акцентируя внимание на внутреннюю структуру, в то время как T-sne и UMAP пытается сохранить ближайшее окружение каждого элемента'
#   # st.balloons()
# # else:
#   # st.error('Ответ неверный!')

# genre3 = form.radio("Чем отличается параметр n_components в алгоритме PCA от всех прочих алгоритмов уменьшения размерности?",
#                  ((''),("Для PCA в качестве параметра n_components выбирается количество кластеров, а в 2х других - в качестве количества измерений"),
#                  ("В алгоритме PCA  параметр n_components используется для определения количества фичей с наибольшей суммарной дисперсией, а в 2х других - для количества кластеров"),
#                  ("В алгоритме PCA параметр n_components используется для определения количества фичей с наибольшей суммарной дисперсией, а в 2х других - для определения количества измерений")))
# genre3_answer = "В алгоритме PCA параметр n_components используется для определения количества фичей с наибольшей суммарной дисперсией, а в 2х других - для определения количества измерений"
# button_test = form.form_submit_button('Пройти тест')
# if button_test:
#   if genre == genre_answer and genre2 == genre2_answer and genre3 == genre3_answer:
#     form.success('Ответ верный!')
#     form.baloons
#   else:
#     form.error('Вы где-то ошиблись!')


with st.form('Ответьте на все вопросы, чтобы успешно завершить лабораторную работу'):
    st.markdown('**Вопрос 1:** Какой алгоритм кластеризации предполагает наличие центроид?')
    question_1_wrong_1 = st.checkbox('Agglomerative clustering', value=False, key='1')
    question_1_right_2 = st.checkbox('KMeans', value=False, key='2')
    question_1_wrong_3 = st.checkbox('И тот, и другой', value=False, key='3')
    question_1_wrong_4 = st.checkbox('Ни тот, и ни другой', value=False, key='4')

    st.markdown('**Вопрос 2:** Чем уменьшитель размерности PCA отличается от двух других, упомянутых')
    question_2_wrong_1 = st.checkbox('PCA в качестве главного инструмента использует построение графа, где каждый элемент связывается с n-ближайших соседей', value=False, key='5')
    question_2_right_2 = st.checkbox('PCA пытается отобразить все кластеры в целом, меньше акцентируя внимание на внутреннюю структуру, в то время как T-sne и UMAP пытается сохранить ближайшее окружение каждого элемента', value=False, key='6')
    question_2_wrong_3 = st.checkbox('PCA в отличае от T-sne и UMAP предназначен для работы лишь с небольшим количеством данных', value=False, key='7')
    question_2_wrong_4 = st.checkbox('Ничем', value=False, key='8')

    st.markdown('**Вопрос 3:** Чем отличается параметр n_components в алгоритме PCA от всех прочих алгоритмов уменьшения размерности?')
    question_3_wrong_1 = st.checkbox('Для PCA в качестве параметра n_components выбирается количество кластеров, а в 2х других - в качестве количества измерений', value=False, key='9')
    question_3_wrong_2 = st.checkbox('В алгоритме PCA  параметр n_components используется для определения количества фичей с наибольшей суммарной дисперсией, а в 2х других - для количества кластеров', value=False, key='10')
    question_3_right_3 = st.checkbox('В алгоритме PCA параметр n_components используется для определения количества фичей с наибольшей суммарной дисперсией, а в 2х других - для определения количества измерений', value=False, key='11')
    # question_3_wrong_4 = st.checkbox('Ни тот, и ни другой', value=False, key='12')
    if st.form_submit_button('Закончить тест и посмотреть результаты'):
        if question_1_right_2 and question_2_right_2 and question_3_right_3:
            st.markdown('''<h3 style='text-align: left; color: green;'
            >Тест сдан! Лабораторная работа завершена.</h3>''', 
            unsafe_allow_html=True) 
        else:
            st.markdown('''<h3 style='text-align: left; color: red;'
            >Тест не сдан! Где-то была допущена ошибка.</h3>''', 
            unsafe_allow_html=True)
