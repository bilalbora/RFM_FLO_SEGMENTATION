###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

# 1. İş Problemi (Business Problem)
# 2. Veriyi Anlama ve Hazırlama (Data Understanding and Preperation)
# 3. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
# 4. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
# 5. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
# 6. Tüm Sürecin Fonksiyonlaştırılması

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.
# Bu e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Değişkenler

# master_id = Eşsiz müşteri numarası
# order_channeL = Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel = En son alışverişin yapıldığı kanal
# first_order_date = Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date = Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online = Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline = Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online = Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline = Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline = Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online = Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 = Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi



###############################################################
# 2. Veriyi Anlama (Data Understanding)
###############################################################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv('flo_data_20k.csv')
df_copy = df.copy()

df.head(10)
df.columns
df.describe().T
df.isnull().sum()
df.dtypes

columns_to_convert =  df.columns[df.columns.str.contains("date")]
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_datetime)

df['order_channel'].value_counts()
ord_channel_ratio = 100 * df['order_channel'].value_counts() / len(df)
print(ord_channel_ratio)

# We clearly see that, about fifty percent of our customers use android app.
df.groupby('order_channel').agg({'total_order': ['sum'],
                                 'total_value': ['sum']})

def dataprep(df):
    #Copying df on df_copy

    ## Creating total_order and total_value variables
    df['total_order'] = df['order_num_total_ever_online'].astype(int) + df['order_num_total_ever_offline'].astype(int)
    df['total_value'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']

    # Converting the date columns
    columns_to_convert = df.columns[df.columns.str.contains("date")]
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_datetime)

    # Our top 10 customers for value and order
    top_10_order = df.groupby('master_id').agg({'total_value': 'sum'}). \
        sort_values(by='total_value', ascending=False).head(10)

    top_10_value = df.groupby('master_id').agg({'total_order': 'sum'}). \
        sort_values(by='total_order', ascending=False).head(10)
    return df


###############################################################
# 3. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
###############################################################

def calculate_rfm_metrics(df):
    # Find the last order date
    df['last_order_date'].max()
    today_date = dt.datetime(2021, 6, 1)

    # Create recency column
    df["recency"] = (today_date - df["last_order_date"]).astype('timedelta64[D]')
    df['recency'] = df['recency'].astype(int)
    # Create frequency column
    df['frequency'] = df['total_order']
    # Create monetary column
    df['monetary'] = df['total_value']

    rfm = pd.DataFrame()
    rfm = df[['master_id', 'recency', 'frequency', 'monetary']]
    return rfm

###############################################################
# 4. RF Skorlarının Hesaplanması (Calculating RFM Scores)
###############################################################

def calculate_rfm_scores(rfm):
    # Create recency score
    rfm['rec_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

    # Create frequency score
    rfm['freq_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

    # Create monetary score
    rfm['mon_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    rfm["RF_SCORE"] = (rfm['rec_score'].astype(str) +
                        rfm['freq_score'].astype(str))
    return rfm

###############################################################
# 5. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
###############################################################

def segmantation(rfm):
# regex

# RFM names
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

    rfm.groupby('segment').agg({'recency': 'mean',
                                'frequency': 'mean',
                                'monetary': 'mean'})

    return rfm

###############################################################
# 7. Tüm Sürecin Fonksiyonlaştırılması
###############################################################

def create_rfm(dataframe, csv=False):
    dataprep(dataframe)
    calculate_rfm_metrics(dataframe)
    calculate_rfm_scores(dataframe)
    segmantation(dataframe)

    if csv:
        dataframe.to_csv('rfm_csv')

    return dataframe

create_rfm(df, csv=True)


# The presence of female customers who spend a lot on the
# advertisement of the new product to be launched and recording them as csv

seg = ['loyal_customer', 'champions']
target_segments = df[df["segment"].isin(seg)]["master_id"]
tar_cust = df[(df["master_id"].isin(target_segments)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
tar_cust.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
tar_cust.shape
tar_cust.head()