import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


data = pd.read_csv("datasets/flo_data_20k.csv")

df = data.copy()

df.head(10)

######## Veri Seti Hikayesi ###############

"""Online ayakkabı mağazası olan FLO müşterilerini
segmentlere ayırıp bu segmentlere göre pazarlama
stratejileri belirlemek istiyor. Buna yönelik olarak
müşterilerin davranışları tanımlanacak ve bu
davranışlardaki öbeklenmelere göre gruplar oluşturulacak"""

"""
Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan) 
olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.
"""

##################################################################################################
# master_id Eşsiz müşteri numarası
# order_channel Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel En son alışverişin yapıldığı kanal
# first_order_date Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
##################################################################################################
#Değiskenlerin listesi
print("Değiskenlerin listesi\n", df.columns)
#Betimsel İstatistik
print("Betimsel_İstatistik\n", df.describe().T)
#Boş Değer Analizi
print("Boş Değer Analizi\n", df.isnull().sum())
#Değişken_Tipleri
print("Değişken_Tipleri\n", df.dtypes)


###########Görev=3##################

"""
Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş 
yaptığını ifade etmektedir. Her bir müşterinin toplam
alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz
"""

df["Total_order"] =  df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["Total_order_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

###########Görev=4##################

"""
Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
"""
df['first_order_date'] = df['first_order_date'].astype('datetime64[ns]')
df['last_order_date'] = df['last_order_date'].astype('datetime64[ns]')
df["last_order_date_online"] = df["last_order_date_online"].astype('datetime64[ns]')
df["last_order_date_offline"] = df["last_order_date_offline"].astype('datetime64[ns]')

###########Görev=5##################
"""
Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
"""
print("Kanala göre müsteri sayısı \n ", df.groupby("order_channel").agg({"master_id" : "count"}).sort_values(by="master_id" ,ascending=False))
print("Toplam alınan ürün sayısı \n ", df.groupby("order_channel").agg({"Total_order" : "sum"}).sort_values(by="Total_order" ,ascending=False))
print("Toplam harcamaların sayısı \n ", df.groupby("order_channel").agg({"Total_order_value" : "sum"}).sort_values(by="Total_order_value" ,ascending=False))

###########Görev=6##################
"""
En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
"""
print ( "En fazla kazancı getiren ilk 10 müşteri \n",
df.groupby("master_id").agg({"Total_order_value" : "sum"}).sort_values(by="Total_order_value",ascending=False).head(10)
)

###########Görev 7##################
"""
En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
"""
print ( "En fazla siparişi ilk 10 müşteri \n",
df.groupby("master_id").agg({"Total_order" : "sum"}).sort_values(by="Total_order",ascending=False).head(10)
)

###########Görev 8##################
"""
Veri ön hazırlık sürecini fonksiyonlaştırınız
"""

def data_preprocessing(dataframe):
    import datetime as dt
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)


    dataframe["Total_order"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["Total_order_value"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]

    dataframe['first_order_date'] = dataframe['first_order_date'].astype('datetime64[ns]')
    dataframe['last_order_date'] = dataframe['last_order_date'].astype('datetime64[ns]')
    dataframe["last_order_date_online"] = dataframe["last_order_date_online"].astype('datetime64[ns]')
    dataframe["last_order_date_offline"] = dataframe["last_order_date_offline"].astype('datetime64[ns]')

    return dataframe

###########Görev 9##################
"""
Recency, Frequency ve Monetary tanımlarını yapınız.
Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
"""

"""
recency : Müşterinin alışveriş yaptığı son tarihten bu güne kadar geçen süre.(gün)
frequency : Müşterinin yaptığı alışverişlerin sıklığı.
monetary : Müşterinin yaptığı alışverişlere harcadığı toplam tutar
"""

df["last_order_date"].max()


today_date = dt.datetime(2021, 6, 1)
type(today_date)

rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'Total_order': lambda Total_order: Total_order.sum(),
                                     'Total_order_value': lambda Total_order_value : Total_order_value.sum()})

rfm.reset_index()

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.head()

###########Görev 10: RF Skorunun Hesaplanması##################
"""
Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz. 
Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz. 
"""

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1] )

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))


###########Görev 11: RF Skorunun Segment Olarak Tanımlanması##################

"""
Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.
"""

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

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm.head()

###########Görev 12: Yorumlama ##################

""""
Adım 1 :Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
Adım 2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.
a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz.

b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz


"""

rfm.groupby("segment").agg({"recency" : "mean",
                            "frequency" : "mean",
                            "monetary" : "mean"})

rfm.reset_index(inplace=True)

new_df1 = pd.DataFrame(rfm[(rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")]["master_id"] )

new_df2 = pd.DataFrame(df[df["interested_in_categories_12"].str.contains("KADIN")]["master_id"])

new_df = pd.DataFrame()
new_df["new_customer_id"] = pd.concat([new_df1,new_df2],ignore_index=True)

new_df.to_csv("loyal_woman_cus.csv")


####################################################

new_df1_ = pd.DataFrame(rfm[(rfm["segment"] == "cant_loose") | (rfm["segment"] == 'about_to_sleep') | (rfm["segment"] == 'new_customers')]["master_id"])

new_df2_ = pd.DataFrame(df[(df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("COCUK"))]["master_id"])

new_df_ = pd.DataFrame()
new_df_["new_customer_id"] = pd.concat([new_df1_,new_df2_],ignore_index=True)

new_df.to_csv("attention_boy_man_cus.csv")
