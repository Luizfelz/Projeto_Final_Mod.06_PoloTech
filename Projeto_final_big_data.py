from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from fastapi import FastAPI, Depends
from pyspark.sql import Row
from enum import Enum
from pydantic import BaseModel
import pandas as pd
import numpy as np
import random

spark = SparkSession.builder \
    .master('local[*]') \
    .appName("Projeto Amazon Videogames") \
    .getOrCreate()

# path do arquivo utilizado (dataset)
path = 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Video_Games_v1_00.tsv.gz'

# dataset original
dados = spark.read.format("csv")\
                .option("header","true")\
                .option("inferSchema", "true")\
                .option("delimiter", "\t")\
                .option("multiLine", "true")\
                .option("quote","\"")\
                .option("escape","\"")\
                .csv(path)

# dataset sem as colunas desnecessárias
dados_2 = dados.select(['customer_id', 'review_id', 'product_id',
              'product_parent', 'product_title', 'star_rating'])

# antigo COSTUMERS_10_MAIS: usuários que fizeram mais de 10 reviews
costumers_10_reviews = dados.groupBy('customer_id').agg(f.count('*').alias('count_reviews')).filter('count_reviews >= 10').orderBy('count_reviews', ascending=True)  

# antigo PRODUCTS 5 MAIS: produtos com mais de 5 reviews feitas
products_5_reviews = dados.groupBy('product_title').agg(f.count('*').alias('count_reviews')).filter('count_reviews >= 5').orderBy('count_reviews', ascending = True)

# Selecionando apenas os dados cujos produtos tem mais de 5 reviews
products_5_reviews = products_5_reviews.withColumnRenamed("count_reviews", "product_count_reviews")

# Join datasets
merged_data = dados.join(products_5_reviews, on="product_title", how="inner").select(dados_2["*"], f.col("product_count_reviews"))

# Selecionando apenas os dados cujos usuários fizeram mais de 10 reviews
costumers_10_reviews = costumers_10_reviews.withColumnRenamed("count_reviews", "product_count_reviews")

# Join datasets
merged_data_2 = dados.join(costumers_10_reviews, on="customer_id", how="inner").select(dados_2["*"], f.col("product_count_reviews"))

# dividindo o treino e teste
(training, test) = merged_data_2.randomSplit([0.8, 0.2], seed = 42)

# configurando o modelo
als = ALS( userCol="customer_id", itemCol="product_parent", ratingCol="star_rating",
 coldStartStrategy="drop", nonnegative = True, implicitPrefs = False, seed = 42)

# treina o modelo com o dataset "training"
model = als.fit(training) 
#_____________________
# Faz as predições
predictions = model.transform(test)

# cria o "evaluator" para retornar o Erro Médio Quadrático
evaluator = RegressionEvaluator(metricName="rmse", labelCol="star_rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions) # variável que armazena o erro médio quatrático

# Gera o top 10 recomendações de produtos para cada usuário
user_recs = model.recommendForAllUsers(10)

# Retorna um dataset com somente o Customers_ID e os ID's dos produtos recomendados para cada customer
user_recs_only_item_id = user_recs.select(user_recs['customer_id'], user_recs['recommendations']['product_parent'])

# Cria uma lista com todos os usuários únicos
customer_ids = user_recs_only_item_id.select('customer_id').distinct().collect()

item_recs = model.recommendForAllItems(10)

# # Retorna um dataset com somente as colunas selecionadas
user_check = merged_data_2.select(['customer_id', 'product_id', 'product_parent', 'product_title'])

# # Variável que armazena somente o "product_parent" e os usuários recomendados
item_recs_only_customer_id = item_recs.select(item_recs['product_parent'], item_recs['recommendations']['customer_id'])

# # Lista com os "product_parents" únicos
product_parents = item_recs_only_customer_id.select('product_parent').distinct().collect()

# class para usar na primeira função do predict_Game()
class ModelName1(str, Enum):
    w1 = customer_ids[0].customer_id
    w2 = customer_ids[1].customer_id
    w3 = customer_ids[2].customer_id
    w4 = customer_ids[3].customer_id
    w5 = customer_ids[4].customer_id
    
class Game(BaseModel):
    Customer_ID: ModelName1 = ModelName1.w1

# class para usar na segunda função do predict_Client()
class ModelName2(str, Enum):
    v1 = product_parents[0].product_parent
    v2 = product_parents[1].product_parent
    v3 = product_parents[2].product_parent
    v4 = product_parents[3].product_parent
    v5 = product_parents[4].product_parent
    
class Client(BaseModel):
    Product_ID: ModelName2 = ModelName2.v1

# cria a aplicação FastAPI
app = FastAPI()

# define o endpoint para a previsão
@app.get("/predict_Game")
def predict_Game(Customer_ID: Game=Depends()):
    Customer_ID = str(Customer_ID)
    valor = Customer_ID.split(":")[1].strip(" '>")
  
    # transforma a entrada em um array numpy
    random_customer_id = int(valor)
    
     # # cria um dataframe com somente o usuário escolhido aleatoriamente e seuas respectivas recomendações
    random_customer_df = user_recs_only_item_id.filter(user_recs_only_item_id.customer_id == random_customer_id)
    
    random_customer_pd = random_customer_df.toPandas()
    
    random_customer_pd.iloc[:1, 1]
    result_list = random_customer_pd.iloc[:1, 1].tolist()
    result_list = result_list[0]
    
    # # Retorna, do dataset original, as informações referentes aos ids dos produtos da lista "result_list"
    user_recommendations_df = dados.filter(f.col("product_parent").isin(result_list))
    
    # # Variável que guarda os nomes dos produtos recomendados para o customer_id aleatório
    recomendacao_usuario_aleatorio = user_recommendations_df.select('product_title').distinct().toPandas()


    # retorna a espécie prevista como uma resposta da API
    return {"Recomendação de produtos": recomendacao_usuario_aleatorio}

# define o endpoint para a previsão
@app.get("/predict_Client")
def predict_Client(Product_ID: Client=Depends()):
    Product_ID = str(Product_ID)
    valor = Product_ID.split(":")[1].strip(" '>")
    
    # transforma a entrada em um array numpy
    random_product_id = int(valor)
    # # cria um dataframe com somente o produto escolhido aleatoriamente e seus respectivos usuários recomendados
    random_product_df = item_recs_only_customer_id.filter(item_recs_only_customer_id.product_parent == random_product_id)

    # # cria um Pandas dataframe
    random_product_pd = random_product_df.toPandas()

    # # transforma somente a coluna em uma lista
    random_product_pd.iloc[:1, 1]
    result_list_item = random_product_pd.iloc[:1, 1].tolist()

    # # armazena o primeiro ID
    result_list_item = result_list_item[0]

    # # Retorna um dataframe contendo as informações sobre aquele produto
    item_recommendations_df = dados.filter(f.col("customer_id").isin(result_list_item))

    # # Variável que guarda os ids dos usuários recomendados para o o produto aleatório
    recomendacao_produto_aleatorio = item_recommendations_df.select('customer_id').distinct().toPandas()
    recomendacao_produto_aleatorio = recomendacao_produto_aleatorio.values.tolist()

    # retorna a espécie prevista como uma resposta da API
    return {"Recomendação de clientes (ID's dos clientes)": recomendacao_produto_aleatorio}