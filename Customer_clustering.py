# импортируем библиотеки

from email import header
import streamlit as st
from PIL import Image
import visualiser as vi
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import plotly.graph_objects as go
import plotly.express as px

st.markdown('''<h1 style='text-align: center; color: black;'
            >Составление профилей клиентов посредством кластеризации</h1> 
            \n<h1>Часть 1: знакомство с методами и инструментами</h1>''', 
            unsafe_allow_html=True)
img_meme_1 = Image.open("7qz.jpg")
st.image(img_meme_1,use_column_width='auto')
st.write("""
Данный стримлит предназначен для ознакомления с самой распространённой задачей обучения без учителя, а именно - задачей кластеризации. Основной задачей 
кластеризации является разбиение совокупности объектов на однородные группы и последующего поиска изначально неочевидных, скрытых закономерностей.
\nЦелью кластеризации является получение нового знания из предоставленных данных. Список прикладных областей, в которых применяется 
кластеризация включает в себя: 
\n * сегментацию изображений, 
\n * анализ текстов, 
\n * маркетинг, 
\n * защиту от фрода и многое другое. 
\nЧасто кластеризация является первым шагом при анализе данных. Помимо этого, целью кластеризации может также быть сжатие данных, и уменьшение изначальной выборки путём 
вычленения из каждого кластера наиболее типичных представителей, а также выявление шума - объектов, которые не подходят ни к одному кластеру.
\nВ анализе данных покупателей кластеризация может помочь 
лучше узнать целевую аудиторию, чтобы в дальнейшем опираться на интересы этих выявленных групп, находя индивидуальный подход к каждому клиенту, что 
в свою очередь должно привести к повышению лояльности клиента, и, как следствие - **прибыли**. Такой тип работы с данными клиентов, целью которых является 
выявление освновных своих клиентов и называется [профайлинг](https://ru.wiktionary.org/wiki/%D0%BF%D1%80%D0%BE%D1%84%D0%B0%D0%B9%D0%BB%D0%B8%D0%BD%D0%B3#:~:text=%D0%97%D0%BD%D0%B0%D1%87%D0%B5%D0%BD%D0%B8%D0%B5,%D0%BF%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%D0%BE%D0%B2%2C%20%D0%BD%D0%B5%D0%B2%D0%B5%D1%80%D0%B1%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B3%D0%BE%20%D0%B8%20%D0%B2%D0%B5%D1%80%D0%B1%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B3%D0%BE%20%D0%BF%D0%BE%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D1%8F.). Также, данные, полученные при помощи кластеризации, можно превратить в 
рекомендательную систему, не раздражая клиентов рассылками, но предлагая только те товары, которые им действительно нужны.  

\nПолезно почитать: **[Статья на сайте machinelearning](http://www.machinelearning.ru/wiki/index.php?title=Кластеризация)**, **[Статья на хабре](https://habr.com/ru/company/ods/blog/325654/)**

Стримлит "Кластеризация" состоит из 2 частей:
\n**Первая часть**: с разными методами и способами кластеризации 
\n**Вторая часть**: ознакомление с различными метриками и способами определения оптимального количества кластеров, а также прохождение лабораторной работы

\nДанные подготовили сотрудники ЛИА РАНХиГС.
""")
st.markdown('''<h2 style='text-align: left; color: black;'
            >Пайплайн лабораторной работы:</h2>''', unsafe_allow_html=True)
img_pipeline = Image.open('1_rename.png') #
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
\n**4. Настройка параметров:** составители лабораторной работы провели настройку модели
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
\n**Маленькие_дети** - Количество маленьких детей у клиента на примере 2 популярных учебных датасетов
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


st.subheader('Начало работы')

df = pd.read_csv('customer_clustering.csv')
st.write("""
Теперь давайте представим себя на месте маркетолога, на руках у которога есть собранные о покупателях данные. Наша задача -
провести анализ этих данных с целью выявления и описания групп пользователей. В первой части, как уже говорилось, мы рассмотрим различные
типа кластеризации, а также способы сжатия размерности, и увидим, какие шаги необходимо предпринять для того, чтобы провести "профайлинг" нашей клиентской базы.
Начнём с анализа и очистки данных.
""")

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
missed_values = pd.DataFrame(df.isnull().sum().sort_values(ascending=False), columns=['Количество пропущенных значений'])
if non_val:
  # st.write(pd.DataFrame(df.isnull().sum().sort_values(ascending=False), columns=['количество пропущенных значений']))
  fig = go.Figure(data=[go.Table(
                        header = dict(values = ['',missed_values.columns],
                        line_color='black',
                                            fill_color='#d9d9d9',
                                            align=['center']*len(missed_values.columns),
                                            font=dict(color='black', size=14),
                                            height=30),
                        cells = dict(values =  [missed_values.index,missed_values.values]),
                        columnwidth=[0.4,0.6]
  )])
  fig.update_traces(cells_line_color='black')
  fig.update_layout(width=700, height=245, margin=dict(b=0, l=0, r=1, t=0))
  st.plotly_chart(fig)

#-----------------Preprocessing ---------------

st.subheader('Предварительная обработка данных')

st.write('Давайте сначала обработаем наши данные')

# with st.expander('Первый шаг'):
#with st.form(key='customer_for'):
if st.checkbox('Первый шаг: удаление строк с пропусками'):
  # st.markdown('''
  # Для начала изабавимся от пропущенных значений.
  # ''')
  # if st.checkbox('Удалить пропуски'):
  df = df.dropna()
  st.write(f"Было строк: 2240. Осталось строк: {len(df)}")

if st.checkbox('Второй шаг: определяем, сколько дней человек является нашим клиентом'):
  st.markdown('''Мы привели столбец "Регистрация" к типу данных datetime и посмотрели, когда была первая запись, а когда - последняя. Затем создали новый столбец "Дней_в_магазине". Чтобы получить этот 
  столбец, мы вычли последнюю дату "регистрации" в данных от даты регистрации у каждого клиента. 
  \nТак мы получили информацию, сколько дней человек является нашим клиентом:''')
  # if st.checkbox('Поменять тип данных и получить информацию'):
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

  st.write(f"Самая первая запись клиента: {min(dates)}")
  st.write(f"Самая последняя запись клиента: {max(dates)}")
  # st.dataframe(df['Дней_в_магазине'])
  fig = go.Figure(data=[go.Table(
                              # columnorder = [1], # n for n in range(1, len(df.columns)+1)
                              columnwidth = [0.4,0.6], # *len(title_subsample.columns),
                              header=dict(values=["","Дней в магазине"],
                                          line_color='black',
                                          fill_color='#d9d9d9',
                                          align=['center']*len(df["Дней_в_магазине"]),
                                          font=dict(color='black', size=14),
                                          height=30),                                            
                              cells=dict(values=[df["Дней_в_магазине"].index,df["Дней_в_магазине"]]                                          ))
                            ])
  fig.update_traces(cells_line_color='black')
  fig.update_layout(width=700, height=245, margin=dict(b=0, l=0, r=1, t=0)) # bottom, left, right и top - отступы     title='Наша таблица', title_x=0.5, title_y=1,
  st.plotly_chart(fig)

  
 
if st.checkbox('Третий шаг: делаем новые характиристики из исходных данных'):
  st.write('''**Сейчас мы сделали еще несколько преобразований и создали новые производные столбцы:**
  \n Столбец "Возраст": вычли 2014 из года рождения, чтобы высчитать возраст клиента
  \n Столбец "Покупки": просуммировали покупки по всем категориям
  \n Столбец "С_кем_живет": в зависимости от семейного статуса будет указывать с партнером или один
  \n Столбец "Всего_детей": Посчитали общее количество детей (дети + подростки)
  \n Столбец "Размер_семьи": просуммировали информацию по столбцам "все_дети" и "с_кем_живет"
  \n Столбец "Родитель": 1 - если есть дети, 0 - если нет
  \n Столбец "Образование" соберали в 3 категории: студент (еще учатся), выпускник, магистр/аспирант
  ''')
  # if st.checkbox('Применить преобразования'):
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
    col1, col2 = st.columns([1, 1])
    with col1:
      st.plotly_chart(box_one, use_container_width=True)
    with col2:
      st.plotly_chart(box_two, use_container_width=True)
    st.markdown('''
    Мы оказались правы, есть серьезные выбросы в наших данных. Давайте удалим данные клиента с доходом 666,666k и трех клиентов, которым больше 110 лет"
    ''')
    if st.checkbox('Удалить данные'):
      df = df[(df["Возраст"]<90)]
      df = df[(df["Доход"]<600000)]
      st.write('Вот так теперь выглядят ящики с усами')
      col1, col2 = st.columns([1, 1])
      with col1:
        st.plotly_chart(px.box(df, y='Возраст'), use_container_width=True)
      with col2:
        st.plotly_chart(px.box(df, y='Доход'), use_container_width=True)
      
      



#-----------------Reductor Visualization---------------

st.subheader('Интструменты снижения размерностей')
st.write("""
Теперь, прежде чем говорить о кластеризации, необходимо рассказать об алгоритмах понижения размерности. 
Перед тем, как кластеризировать объекты, нам необходимо перевести их в пространство меньшей размерности (2-ух или 3-ех мерной).
На самом деле понижение размернорности применяют и для других целей (визуализация данных, предобработка признаков перед обучением для борьбы с зашумленными данными), 
но нас сейчас это не интересует.

\n Использование алгоритмов кластеризации на датасете с большим количеством фичей редко приносит нужный результат, что демонстрируется как метриками (о которых 
мы поговорим во второй части), так визуализацией данных. Поэтому для использования алгоритмов кластеризации предварительно используются алгоритмы понижения размерности.
\n Алгоритмы понижения размерности можно разделить на 2 основные группы: 
они пытаются сохранить либо глобальную структуру данных, либо локальные расстояния между точками. К первым относятся такие алгоритмы как [PCA](https://wiki.loginom.ru/articles/principal-component-analysis.html)
(Метод главных компонент), ко вторым [t-SNE](https://habr.com/ru/post/267041/) и [UMAP](https://habr.com/ru/company/newprolab/blog/350584/). 
\n Все визуализации работы алгоритмов уеньшения размерности производятся при помощи кластеризатора KMeans
\n *В графах ниже, где предлагается выбрать числовые параметры для моделей, изначально выставлены дефолтные параметры алгоритмов, **кроме количества кластеров**! Если вам захочется улучшить метрики,
вы можете внести свои коррективы, посмотреть, как будет работать алгоритм, если вы увеличите, или уменьшите тот или иной параметр.*
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
    
    """)
    # form = st.form("form")
    # p = form.number_input("Выберите количество компонентов", min_value = 2, max_value = 28)
    # button_ncomponents = form.form_submit_button("Выявить оптимальное количество компонентов")
    # if button_ncomponents:
    #   visual = vi.Visualise()
    #   st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
    #   st.pyplot(visual.n_component(n_component = p,data = dataframe))

    form =  st.form("Form")
    p = form.number_input("Выберите кoличество компонентов ",min_value = 2,max_value =28,value=2)
    h = form.number_input("Выберите количество кластеров ",min_value = 2,max_value= 15)
    button_pca = form.form_submit_button("Построить график для уменьшителя размерности PCA")
    if button_pca:
        visual = vi.Visualise()
        # if p>=1.00:
            # p = int(p)
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        visual.reductor_vis(reductor = 'PCA',n_clusters = h,n_components = p,data = dataframe)


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
    Большие наборы данных обычно требуют большего значения этого параметра. По умолчанию показатель параметра perplexity установлен на 30.
    
    """)
    form =  st.form("Form")
    comp_input = form.number_input("Выберите кoличество компонентов, до которых будет сжато пространство",min_value = 1,max_value =3,value = 2)
    perp_input = form.number_input("Выберите количество ближайших соседей",min_value = 1.00, max_value = 50.00,value = 30.00)

    h = form.number_input("Выберите количество кластеров",min_value = 2,max_value= 15)
    button_pca = form.form_submit_button("Построить график для уменьшителя размерности t-SNE")
    if button_pca:
        visual = vi.Visualise()
        
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        visual.reductor_vis(reductor = 'TSNE',n_clusters = h,n_components = comp_input,perplexity = perp_input,data = dataframe)
  
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
    #  \n Параметр min_dist определяет, насколько плотно UMAP может упаковывать точки вместе. Это буквально означает минимальное расстояние друг от друга, на 
    # котором точки могут находиться в низкоразмерном представлении. Это означает, что низкие значения min_dist приведут к более громоздким построениям. 
    # Это может быть полезно, если вы заинтересованы в кластеризации или более тонкой топологической структуре. Большие значения min_dist предотвратят объединение 
    # точек UMAP и вместо этого сосредоточатся на сохранении широкой топологической структуры.
    # \n *по дефолту значение min_dist = 0.1, в диапазоне от 0.00 до 0.99*
    expander_bar.markdown(
    """
    \n n_components - количество измерений, до которого снижается размерность датасета
    \n n_neighbours - Этот параметр управляет тем, как UMAP уравновешивает локальную и глобальную структуру данных. Это достигается за счет ограничения 
    размера локальной окрестности, на которую UMAP будет смотреть при попытке изучить многообразную структуру данных. Это означает, что низкие значения 
    n_neighbors заставят UMAP сосредоточиться на очень локальной структуре 
    (потенциально в ущерб общей картине), в то время как большие значения заставят UMAP рассматривать более крупные 
    окрестности каждой точки при оценке многообразной структуры данных. потеря мелкой детализированной структуры ради получения более широких данных.
    \n *по дефолту значение n_neighbours = 15*
   
    """)
    form =  st.form("Form")
    comp_input = form.number_input("(n_components) Выберите кoличество компонентов,до которых будет сжато пространство",min_value = 1,max_value =3,value = 2)
    neigh_input = form.number_input("(n_neighbors) Выберите количество соседей для каждого элемента",min_value = 1, max_value = 100,value = 15)
    # dist_input = form.number_input("(min_dist)Выберите минимальное расстояние",min_value = 0.00,max_value= 0.99)
    cl_input = form.number_input("Выберите количество кластеров",min_value = 2,max_value= 15)

    button_pca = form.form_submit_button("Построить график для уменьшителя размерности UMAP")
    if button_pca:
        visual = vi.Visualise()
        st.set_option('deprecation.showPyplotGlobalUse', False) #hide warning
        visual.reductor_vis(reductor = 'UMAP',n_clusters = cl_input,n_components = comp_input,n_neighbors = neigh_input,data = dataframe)


#--------------------------------------Clustering-----------------------------------------------------

dataframe = pd.read_csv('final_customer_clustering_drop.csv')
st.subheader('Кластеризаторы')
st.write("""
Начать разговор о задачах кластеризации стоит собственно с алгоритмов кластеризации. Ниже представленны те алгоритмы, 
которые автору показались наиболее часто используемыми, а именно [KMeans](https://wiki.loginom.ru/articles/k-means.html) и [Agglomerative Clustering](https://machinelearningmastery.ru/machine-learning-algorithms-part-12-hierarchical-agglomerative-clustering-example-in-python-1e18e0075019/)
. Об их работе в сжатом виде можно прочесть ниже, более подробную информацию можно получить перейдя по ссылке
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
        visual.proto_vis(cluster = 'KMeans',n_clusters = p,data = dataframe)

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
        visual.proto_vis(cluster = 'AgglomerativeClustering',n_clusters = p,data = dataframe)


with st.form('Ответьте на все вопросы, чтобы успешно завершить лабораторную работу'):
    st.markdown('**Вопрос 1:** Для каких задач применяется кластеризациия?')
    question_4_wrong_1 = st.checkbox('Для задач регрессионого характера, в которой необходимо предсказание какого-то числового значения',value = False,key = 12)
    question_4_right_2 = st.checkbox('Для разбиения совокупности объектов на однородные группы',value = False,key = 13)
    question_4_wrong_3 = st.checkbox('Для очистки данных',value = False, key = 14)

    st.markdown('**Вопрос 2:** Для чего перед кластериацией используется уменьшитель размерности?')
    question_5_wrong_1 = st.checkbox('Для устранения всех "выбросов" в данных', value = False, key = 15)
    question_5_wrong_2 = st.checkbox('Для того, чтобы уменьшить количество данных',value = False,key = 16)
    question_5_right_3 = st.checkbox('Для того, чтобы улучшить данные кластеризации',value = False,key = 17)

    st.markdown('**Вопрос 3:** Какой алгоритм кластеризации предполагает наличие центроид?')
    question_1_wrong_1 = st.checkbox('Agglomerative clustering', value=False, key='1')
    question_1_right_2 = st.checkbox('KMeans', value=False, key='2')
    question_1_wrong_3 = st.checkbox('И тот, и другой', value=False, key='3')
    question_1_wrong_4 = st.checkbox('Ни тот, и ни другой', value=False, key='4')

    st.markdown('**Вопрос 4:** Чем уменьшитель размерности PCA отличается от двух других, упомянутых')
    question_2_wrong_1 = st.checkbox('PCA в качестве главного инструмента использует построение графа, где каждый элемент связывается с n-ближайших соседей', value=False, key='5')
    question_2_right_2 = st.checkbox('PCA пытается отобразить все кластеры в целом, меньше акцентируя внимание на внутреннюю структуру, в то время как T-sne и UMAP пытается сохранить ближайшее окружение каждого элемента', value=False, key='6')
    question_2_wrong_3 = st.checkbox('PCA в отличае от T-sne и UMAP предназначен для работы лишь с небольшим количеством данных', value=False, key='7')
    question_2_wrong_4 = st.checkbox('Ничем', value=False, key='8')

    # st.markdown('**Вопрос 4:** Чем отличается параметр n_components в алгоритме PCA от всех прочих алгоритмов уменьшения размерности?')
    # question_3_wrong_1 = st.checkbox('Для PCA в качестве параметра n_components выбирается количество кластеров, а в 2х других - в качестве количества измерений', value=False, key='9')
    # question_3_wrong_2 = st.checkbox('В алгоритме PCA  параметр n_components используется для определения количества фичей с наибольшей суммарной дисперсией, а в 2х других - для количества кластеров', value=False, key='10')
    # question_3_right_3 = st.checkbox('В алгоритме PCA параметр n_components используется для определения количества фичей с наибольшей суммарной дисперсией, а в 2х других - для определения количества измерений', value=False, key='11')
    # question_3_wrong_4 = st.checkbox('Ни тот, и ни другой', value=False, key='12')
    answers=(question_4_wrong_1 or question_4_wrong_3 or question_5_wrong_1 or question_5_wrong_2 or question_1_wrong_1 or question_1_wrong_3 or 
    question_1_wrong_4 or question_2_wrong_1 or question_2_wrong_3 or question_2_wrong_4)
    if st.form_submit_button('Закончить тест и посмотреть результаты'):
        if (question_1_right_2 and question_2_right_2 and question_4_right_2 and question_5_right_3)==True and answers == False:
            st.markdown('''<h3 style='text-align: left; color: green;'
            >Тест сдан! Теперь преступайте к выполнению второй части лабортаной работы.</h3>''', 
            unsafe_allow_html=True) 
        else:
            st.markdown('''<h3 style='text-align: left; color: red;'
            >Тест не сдан! Где-то была допущена ошибка.</h3>''', 
            unsafe_allow_html=True)
img_meme2 = Image.open('1_ravEQiopKoM9RgxUhxbP1Q.jpeg')
st.image(img_meme2,width = 400,caption = "В нашем стримлите мы пытаемся разосновать посыл этой шутки")#use_column_width='auto')
# col1,col2,col3=st.columns(3)
# with col1:
#   st.write('')
# with col2:
#   st.image(img_meme2,width = 400)#use_column_width='auto')
# with col3:
#   st.write('')