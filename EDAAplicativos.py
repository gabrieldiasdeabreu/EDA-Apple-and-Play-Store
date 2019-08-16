#!/usr/bin/env python
# coding: utf-8

# # Análise dos dados da Apple Store e Play Store 
# ## Gabriel Dias de Abreu
# ## Sistemas Apoio à decisão - UFJF 2019.1
# ### 23/04/2019

# # Objetivo do Estudo:
# 
# # Gerar conhecimento para o contexto de um desenvolvedor mobile adentrando o mercado.
# 
# 
# 
# 
# 

# ## Dessa forma,
# 
# ### Analizando as seguintes features das duas prinicipais lojas de aplicativos mobile
# + Categoria - classificação do aplicativo por parte da loja, classes mostradas adiante nos histogramas (discreto e  qualitativo).
# + Rating - classificação do aplicativo na loja, atributo discreto no conjunto [0, 1, 2, 3, 4, 5],
# + Size - tamanho em Bytes do aplicativo, foi utilizado para indicar o quanto um aplicativo foi bem trabalhado, [o, inf] contínuo
# + Price - preço em dólares para aplicativos comprados pela loja, contínuo [0, 400],
# + Reviews - quantidade de usuários que avaliaram o aplicativo, contínuo [0, inf],
# + Content Rating - faixa etária para o aplicativo, discreto, apresentado no histograma

# # Features Retiradas da base de dados da Apple Store 
# ### {'ver', 'ipadSc_urls.num', 'currency', 'Unnamed: 0', 'lang.num', 'rating_count_ver', 'sup_devices.num', 'id', 'user_rating_ver', 'vpp_lic'}
# # Features Retiradas da base de dados da Play Store 
# ### {'Type', 'Genres', 'Current Ver', 'Android Ver', 'Installs', 'Last Updated'}

# In[1]:


get_ipython().run_cell_magic('html', '', '<style>\ndiv.input {\n    display:none;\n}\n</style>')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2


# In[3]:


dadosPlayStore = pd.read_csv('google-play-store-apps/googleplaystore.csv')
dadosAppleStore = pd.read_csv('google-play-store-apps/AppleStore.csv')


# In[4]:


def plotaVenn():
    plt.figure(figsize=[10, 8], dpi=200)
    venn2([set(dadosPlayStore['App']), set(dadosAppleStore['track_name'])], ['Apps Play Store', 'Apps Apple Store'])
    plt.show()


# ## Quantidade de Apps em cada loja e intersecção 

# In[5]:


plotaVenn()
# print('quantidade de APPs e intersecção')


# In[6]:


def plotaQtdApps():    
#     print('total de apps', len(dadosAppleStore) + len(dadosPlayStore))
    plt.figure(figsize=[10, 8], dpi=200)
    plt.pie([len(dadosAppleStore), len(dadosPlayStore)], explode=[0.1, 0], labels=['Apple Store', 'Google Store'], autopct='%1.1f%%', shadow=True, startangle=140)
    # plt.legend()
    
#     plt.show()
    plt.show()


# In[7]:


plotaQtdApps()


# In[8]:


listaInteressantesPlayStore = ['App', 'Category', 'Rating', 'Size', 'Price', 'Reviews',  'Content Rating']
listaInteressantesAppleStore = ['track_name','prime_genre', 'user_rating', 'size_bytes', 'price', 'rating_count_tot', 'cont_rating', ]


# In[9]:


print(set(dadosAppleStore.keys()).difference(listaInteressantesAppleStore))
print(set(dadosPlayStore.keys()).difference(listaInteressantesPlayStore))


# In[10]:


dadosAppleStore = dadosAppleStore[listaInteressantesAppleStore]
dadosPlayStore = dadosPlayStore[listaInteressantesPlayStore]
print(dadosAppleStore.keys())
print(dadosPlayStore.keys())


# # Categorias Dos Aplicativos
# ## Categorias dos aplicativos e sua quantidade
# 
# # Qual categoria de aplicativo faz mais sucesso no universo mobile para um desenvolvedor mobile tomar como foco e publicar?

# In[11]:


def plotaQtdAppsCategoriasGoog():
    # print(dadosAppleStore.keys(), dadosPlayStore.keys())
    plt.figure(figsize=[10,8], dpi=200)
#     idx =plt.subplot(121)
    plt.title('Google Store')
    dadosPlayStore.groupby('Category')['App'].count().plot.barh()
    plt.ylabel('Categoria')
    plt.xlabel('Quantidade')
    plt.grid(b=True, axis='x')
    plt.show()
#     plt.subplot(122, sharex=idx)
#     plt.figure(figsize=[20,18], dpi=120)
def plotaQtdAppsCategoriasAppl():
    plt.figure(figsize=[10,8], dpi=200)
    plt.title('Apple Store')
    dadosAppleStore.groupby('prime_genre')['track_name'].count().plot.barh()
    plt.ylabel('Categoria')
    plt.xlabel('Quantidade')
    plt.grid(b=True, axis='x')
    plt.show()


# In[12]:


plotaQtdAppsCategoriasGoog()


# In[13]:


plotaQtdAppsCategoriasAppl()


# # As duas lojas tem como grande parte das publicações Jogos sendo portanto uma categoria promissora
# 
# ## uma segunda categoria seriam aplicativos educativos e de família

# # Observações 
# + Os jogos parecem ser uma tendência nos celulares
# + Removido o registro com problemas no rate e na categoria
# + App de Família: "Designed for Families expands the visibility of your family content on Google Play, helping parents easily find your family-friendly apps and games throughout the store. Other features create a trusted environment that empowers parents to make informed decisions and engage with your content." https://developer.android.com/google-play/guides/families

# In[14]:


dadosPlayStore = dadosPlayStore[dadosPlayStore.Rating < 10]


# In[15]:


print('numero de aplicativos com o nome repetido:', len(set(dadosPlayStore.App).intersection(dadosAppleStore.track_name)))


# # Tamanho Aplicativos
#  ## Problema:
#  ## Os dados da Play store estao em 2.2MB enquanto a apple em 2200000B
#  ## 17% dos dados de tamanho do aplicativo da play store estão como "variaveis com dispositivo"
#  
#  + imputando os dados que tem "Variavel com dispositivo" como a mediana da categoria;
#  + transformando os dados da Play Store em bytes

# # Observando a qualidade ou reserva de mercado pelo tamanho dos aplicativos. Qual seria a melhor loja para começar a publicar e testar um MVP?

# In[16]:


filtroBytes = lambda x : np.nan if x == 'Varies with device' else int(float(x.split('M')[0])*1000000) if x[-1] == "M" else int(float(x.split('k')[0])*1000)
# print()[filtro(i) for i in dadosPlayStore.Size].count(np.nan)/len(dadosPlayStore.Size)
dadosPlayStore.Size = list(map(filtroBytes, dadosPlayStore.Size))
# [print(filtro(i),j) for i, j in zip(dadosPlayStore.Size[dadosPlayStore.Size != 'Varies with device'], dadosAppleStore.size_bytes)]
# print(dadosPlayStore)
# [filtroImputaNan(size, category) for size, category in zip(dadosPlayStore.Size, dadosPlayStore.Category)]
# dadosPlayStore.Size
# dadosPlayStore.apply(teste)


# In[17]:


medianasCategorias = dadosPlayStore.groupby('Category').median()['Size']

filtroImputaNan = lambda x, categoria: medianasCategorias.loc[categoria] if np.isnan(x)  else x
# teste = lambda x: print('oi', x)
# dadosPlayStore.apply(filtroImputaNan, axis=1)
# dadosPlayStore.Size
dadosPlayStore.Size = [filtroImputaNan(size, category) for size, category in zip(dadosPlayStore.Size, dadosPlayStore.Category)]


# In[18]:


def plotaDistribuicaoTamanhos():
    plt.figure(figsize=[10, 8], dpi=200)
    # sns.distplot(dadosPlayStore.Rating.dropna(), label='Google Play Store')
    # sns.distplot(dadosAppleStore.user_rating.dropna(), label='Apple Store')
    # plt.legend()

    # idx = plt.subplot(221)
    # plt.title('Google Play Store')
    # plt.ylabel('rate')
    # pd.DataFrame(dadosPlayStore.Size).boxplot()

    # plt.subplot(222, sharey=idx)
    # plt.title('Apple Store')
    # pd.DataFrame(dadosAppleStore.size_bytes).boxplot()

    jdx = plt.subplot(211)
    plt.ylabel('kde')
    plt.title('Distribuição tamanho aplicativos Google Store')
    sns.distplot(np.log(dadosPlayStore.Size))
    plt.xlabel('')

    plt.subplot(212, sharey=jdx, sharex=jdx)
    plt.title('Distribuição tamanho aplicativos Apple Store')
    plt.ylabel('kde')

    sns.distplot(np.log(dadosAppleStore.size_bytes))
    plt.xlabel('Logn(Bytes)')
    plt.show()


# In[19]:


plotaDistribuicaoTamanhos()


# # Fica evidente pelas Distribuições em *LOGN* que a PlayStore possui melhores oportunidades para projetos iniciais em validação

# In[20]:


dadosPlayStore['Price'] = list(map(lambda x: float(x[1:]) if x[0] == '$' else 0, dadosPlayStore.Price))
# pd.DataFrame(dadosAppleStore.price).boxplot()


# In[21]:


def plotaPrecoApps():
    plt.figure(figsize=[10, 8], dpi=200)

    # pd.DataFrame({'Apple':dadosAppleStore.price, 'PlayStore':dadosPlayStore['Price']}).plot(kind='kde')

    plt.subplot(121)
    pd.DataFrame({'Apple':dadosAppleStore.price, 'PlayStore':dadosPlayStore['Price']}).boxplot( showfliers=False)
    plt.title('Sem outliers')
    plt.ylabel('Price (USD)')

    plt.subplot(122)
    pd.DataFrame({'Apple':dadosAppleStore.price, 'PlayStore':dadosPlayStore['Price']}).boxplot( showfliers=True)
    plt.title('Com outliers')

    plt.show()
    
    
def plotPrecoAppsPie():
    plt.figure(figsize=[16, 8], dpi=200)
    plt.subplot(121)
    plt.title('Apple Store')
    # sns.distplot()
    contagemApple = [
    len(dadosAppleStore[dadosAppleStore.price == 0]),
    len(dadosAppleStore[(dadosAppleStore.price <= 10) & (dadosAppleStore.price !=0)]),
    len(dadosAppleStore[dadosAppleStore.price > 10])
    ]

    labels = ['grátis', 'até 10 dólares', 'maior que 10 dólares']

    contagemPlayStore = [
    len(dadosPlayStore[dadosPlayStore.Price == 0]),
    len(dadosPlayStore[(dadosPlayStore.Price <= 10) & (dadosPlayStore.Price !=0)]),
    len(dadosPlayStore[dadosPlayStore.Price > 10])
    ]

    explode = [0.0, 0.2, 0.52]

    # plt.legend()
    
    plt.title('Play Store')
    plt.pie(contagemPlayStore, explode=explode, labels=labels, autopct='%1.f%%', shadow=True, startangle=140)
    plt.subplot(122)
    plt.pie(contagemApple, explode=explode, labels=labels, autopct='%1.f%%', shadow=True, startangle=140)

    plt.show()


# # Preços cobrados por aplicativos em cada plataforma
# 
# # Como monetizar o meu aplicativo em cada loja?

# In[22]:


plotaPrecoApps()


# In[23]:


plotPrecoAppsPie()


# # Ambas as lojas parecem adotar um modelo gratuito com propagandas. No entanto, na Apple Store é viável a venda por até 10 dólares sendo comum essa prática e enraizada na cultura dos usuários.

# In[24]:


# sns.distplot(dadosAppleStore.rating_count_tot)


# In[25]:


# print(set(dadosPlayStore['Content Rating']))
dadosPlayStore['Content Rating'][dadosPlayStore['Content Rating'] == 'Everyone' ] = '4+'
dadosPlayStore['Content Rating'][dadosPlayStore['Content Rating'] == 'Everyone 10+' ] = '9+'
dadosPlayStore['Content Rating'][dadosPlayStore['Content Rating'] == 'Mature 17+' ] = '17+'
dadosPlayStore['Content Rating'][dadosPlayStore['Content Rating'] == 'Adults only 18+' ] = '17+'
dadosPlayStore['Content Rating'][dadosPlayStore['Content Rating'] == 'Teen' ] = '12+'
dadosPlayStore['Content Rating'][dadosPlayStore['Content Rating'] == 'Unrated' ] = '4+'
# print(dadosPlayStore[dadosPlayStore['Content Rating'] == 'Unrated' ])

# print(set(dadosAppleStore['cont_rating']))
# print(set(dadosPlayStore['Content Rating']))


# In[26]:


print(set(dadosAppleStore['cont_rating']))
print(set(dadosPlayStore['Content Rating']))


# In[27]:


def plotaFaixaEtaria():
    dictAux = {'AppleStore':dadosAppleStore.groupby('cont_rating').count()['track_name']/ len(dadosAppleStore), 
           'PlayStore':dadosPlayStore.groupby('Content Rating').count()['App']/ len(dadosPlayStore)}

    # plt.figure(figsize=[16, 9])
#     plt.figure(figsize=[16, 9], dpi=96)
    pd.DataFrame(dictAux).plot(kind='barh', figsize=[10, 8])
    plt.grid(b=True, axis='y')
    plt.xlabel('% de apps')
    plt.ylabel('Faixa etária')
    plt.show()


# # Faixa etária
# + Manipulação para ficarem na mesma codificação
# + Um registro tinha a categoria errada, foi corrigido olhando o site

# # Qual público devo privilegiar para cada loja?

# In[28]:


plotaFaixaEtaria()


# # A maioria dos aplicativos são livres para todos os públicos (4+), portanto, aplicativos de propósito geral e livres de faxa etária são os mais viáveis nas duas lojas.
# 
# ## No entanto, aplicativos para maiores de 17 e 9 anos são mais comuns na Apple Store, assim, como os para maiores de 12 anos, porém, de forma mais discreta.
# 

# In[29]:


print(dadosAppleStore.keys())
print(dadosPlayStore.keys())


# In[30]:


import math
def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


# In[31]:


dadosPlayStore['Rating'] = list(map(lambda x: normal_round(x), dadosPlayStore['Rating']))
dadosAppleStore['user_rating'] = list(map(lambda x: normal_round(x), dadosAppleStore['user_rating']))
# print(set(dadosAppleStore['user_rating']))


# In[32]:


def plotaRate():
    dictAux = {'AppleStore':dadosAppleStore.groupby('user_rating').count()['track_name']/ len(dadosAppleStore), 
           'PlayStore':dadosPlayStore.groupby('Rating').count()['App']/ len(dadosPlayStore)}
#     plt.figure(figsize=[16, 9], dpi=96)

    # somaRate = {'AppleStore':dadosAppleStore.groupby('rating_count_tot').count()['track_name']/ len(dadosAppleStore), 
    #            'PlayStore':dadosPlayStore.groupby('Reviews').count()['App']/ len(dadosPlayStore)}

    # (dadosPlayStore['Rating'])
    # sns.distplot(dadosAppleStore['user_rating'])
    pd.DataFrame(dictAux).plot.barh( figsize=[10, 8])
    plt.xlabel('% of apps')
    plt.ylabel('rate')
    plt.title('porcentagem dos apps que têm certa nota')
    # pd.DataFrame(somaRate).plot.bar()
    plt.show()


# # Avaliação pelos usuários
# + foi discretizada para [0, 1, 2, 3, 4, 5]

# # Qual a loja em que terei mais chance de ser bem avaliado?

# In[33]:


plotaRate()


# # Os usuários da Apple Store avaliam mais como 5 mostrando um maior engajamento, talvez devido ao modelo de comprar os apps por valores baixos, entretanto, avaliam mais vezes com 0, e 1 podendo ser um risco para os desenvolvedores. 
# 
# # Os usuários da Play Store tendem a dar nota 4 e 5 na maior parte das vezes, o que configura um cenário mais estável para um produto que queira evitar extremos como nota 0

# In[34]:


dadosPlayStore.Reviews = list(map(lambda x: int(x), dadosPlayStore.Reviews))


# In[35]:


def plotaAvaliacoes():
    sumRate = {'AppleStore':dadosAppleStore.groupby('user_rating')['rating_count_tot'].sum(), 
           'PlayStore':dadosPlayStore.groupby('Rating')['Reviews'].sum()}
#     plt.figure(figsize=[16, 9], dpi=96)

    pd.DataFrame(sumRate).plot.barh(figsize=[10, 8])
    plt.xlabel('Número pessoas')
    plt.ylabel('rate')
    plt.title('Total de pessoas que avaliaram por Rate do Aplicativo')
    plt.show()


# # Número de pessoas que avaliam por Nota

# # Qual o tamanho do engajamento para avaliar meu aplicativo em cada loja?

# In[36]:


plotaAvaliacoes()


# # Os usuários da Play Store são mais engajados em avaliar com notas altas quando identificam a necessidade.

# # Matriz de correlação entre Atributos

# # Existe alguma correlação entre preço e avaliação do aplicativo?
# 
# # ou entre tamanho e avaliação?

# In[37]:


plt.figure(figsize=[10, 8], dpi=90)
sns.heatmap(dadosAppleStore.corr())
plt.title('Apple Store')
plt.show()


# In[38]:


plt.figure(figsize=[10, 8], dpi=90)
sns.heatmap(dadosPlayStore.corr())
plt.title('Play Store')
plt.show()


# # Nenhuma das lojas apresentou uma correlação expressiva, portanto, não parecem ser fatores correlacionados

# # Clusterização da Base de Dados
# 
# # Quais grupos avaliam melhor o aplicativo em cada loja para definir meu público-alvo ?

# In[39]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# In[40]:


dadosAppleStore.keys()


# In[41]:


dadosAppleStore[['prime_genre', 'user_rating', 'size_bytes', 'price', 'rating_count_tot', 'cont_rating']].head()
# encoderOneHot = OneHotEncoder(list(set(dadosAppleStore['prime_genre'])))
# encoderOneHot.fit_transform(dadosAppleStore['prime_genre'])


# In[42]:


filtraEtaria = lambda x: int(x[:-1])
dadosAppleStore['cont_rating'] = list(map(filtraEtaria, dadosAppleStore['cont_rating']))


# In[43]:


dadosAppleStoreCluster = pd.get_dummies(dadosAppleStore['prime_genre'])
dadosAppleStoreCluster['user_rating'] = dadosAppleStore['user_rating']
dadosAppleStoreCluster['size_bytes'] = dadosAppleStore['size_bytes']
dadosAppleStoreCluster['price'] = dadosAppleStore['price']
dadosAppleStoreCluster['rating_count_tot'] = dadosAppleStore['rating_count_tot']
dadosAppleStoreCluster['cont_rating'] = LabelEncoder().fit_transform(dadosAppleStore['cont_rating'] )

dadosAppleStoreCluster[dadosAppleStoreCluster.columns] = MinMaxScaler().fit_transform(dadosAppleStoreCluster[dadosAppleStoreCluster.columns])


# In[44]:


dadosAppleStoreCluster.head()


# ## Tratamento dos dados para aplicar kmeans
# 

# In[45]:


print('Antes')
dadosPlayStore.head()


# In[46]:


dadosPlayStore['Content Rating'] = list(map(filtraEtaria, dadosPlayStore['Content Rating']))


# In[47]:


dadosPlayStoreCluster = pd.get_dummies(dadosPlayStore['Category'])
dadosPlayStoreCluster['Rating'] = dadosPlayStore['Rating']
dadosPlayStoreCluster['Size'] = dadosPlayStore['Size']
dadosPlayStoreCluster['Price'] = dadosPlayStore['Price']
dadosPlayStoreCluster['Reviews'] = dadosPlayStore['Reviews']
dadosPlayStoreCluster['Content Rating'] = LabelEncoder().fit_transform(dadosPlayStore['Content Rating'] )

dadosPlayStoreCluster[dadosPlayStoreCluster.columns] = MinMaxScaler().fit_transform(dadosPlayStoreCluster[dadosPlayStoreCluster.columns])
print('Depois')
dadosPlayStoreCluster.head()


# In[48]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# for i in range(1, 100):
# modelKmeans = KMeans(5)

# print(modelKmeans.score(dadosPlayStoreCluster))


# In[49]:


dadosPlayStoreCluster['kmeans'] = KMeans(5).fit_predict(dadosPlayStoreCluster)
dadosAppleStoreCluster['kmeans'] = KMeans(6).fit_predict(dadosAppleStoreCluster)


# In[50]:


def plotaPca():
    modelPca = TSNE(2)
    dadosReduzidosPlayStore = modelPca.fit_transform(dadosPlayStoreCluster)
    dadosReduzidosAppleStore = modelPca.fit_transform(dadosAppleStoreCluster)
    plt.figure(figsize=[16, 9])
    plt.subplot(1,2,1)
    
    clusterPlayStore = KMeans(8).fit_predict(dadosReduzidosPlayStore)
    plt.scatter(dadosReduzidosPlayStore.T[0], dadosReduzidosPlayStore.T[1], c=clusterPlayStore)

    plt.title('PlayStore scatter')
    # plt.show()
    plt.subplot(1,2,2)
    plt.title('AppleStore scatter')
    clusterAppleStore = KMeans(8).fit_predict(dadosReduzidosAppleStore)
    plt.scatter(dadosReduzidosAppleStore.T[0], dadosReduzidosAppleStore.T[1], c=clusterAppleStore)
    plt.show()


# In[51]:


def plotaPcaM():
    modelPca = PCA(2)
    dadosReduzidosPlayStore = modelPca.fit_transform(dadosPlayStoreCluster)
    dadosReduzidosAppleStore = modelPca.fit_transform(dadosAppleStoreCluster)
    plt.figure(figsize=[16, 9])
    plt.subplot(1,2,1)
    
    clusterPlayStore = KMeans(8).fit_predict(dadosReduzidosPlayStore)
    plt.scatter(dadosReduzidosPlayStore.T[0], dadosReduzidosPlayStore.T[1], c=clusterPlayStore)

    plt.title('PlayStore scatter')
    # plt.show()
    plt.subplot(1,2,2)
    plt.title('AppleStore scatter')
    clusterAppleStore = KMeans(8).fit_predict(dadosReduzidosAppleStore)
    plt.scatter(dadosReduzidosAppleStore.T[0], dadosReduzidosAppleStore.T[1], c=clusterAppleStore)
    plt.show()


# In[52]:


def pairPlotApple():
#     plt.figure(figsize=[16, 9])
    sns.pairplot(dadosAppleStore, height=1.8, diag_kind="kde")
#     plt.title('Apple Store')
    plt.show()


# In[53]:


def pairPlotPlay():
#     plt.figure(figsize=[16, 9])
    sns.pairplot(dadosPlayStore, height=1.8, diag_kind="kde")
#     plt.title('PlayStore')
    plt.show()


# # Play Store Pair Plot

# In[54]:


pairPlotPlay()


# # Apple Store Pair Plot

# In[55]:


pairPlotApple()


# In[56]:


#  redução de dimensão por T-SNE


# In[57]:


# plotaPca()


# In[58]:


# reducao de dimensão por pca


# In[59]:


# plotaPcaM()


# In[60]:


# dadosPlayStoreCluster.groupby('kmeans')[['Rating', 'Size', 'Price', 'Reviews', 'Content Rating']].plot.box()
# plt.title('play Store')
# plt.show()


# In[61]:


def plotaDistClusterApple():
    # plt.subplot(1 ,2 ,1)
    # plt.figure(figsize=[16, 9])
    fig, axes = plt.subplots(2, 3, figsize=[21, 16])
    k, i, j = (0, 0, 0)

    for grupo in dadosAppleStoreCluster.groupby('kmeans'):
        if(i == 2):
    #         if i==0 and j==1:
            i = 0
            j += 1
        k+=1
        grupo[1][['user_rating', 'size_bytes', 'price', 'rating_count_tot', 'cont_rating']].plot.box(ax=axes[i, j], title='AppleStore c '+str(k) )
        i+=1

    # fig.set_title('playStore')
        # plt.subplot(1 ,2 ,2)


    plt.show()


# # Estudo da distribuição dos agrupamentos Formados para a Apple Store

# In[62]:


plotaDistClusterApple()


# # Como pode ser observado:
# + Os grupos que pagaram algum valor ( cluster 1, 2, 3, 6) avaliaram melhor o aplicativo.
# + Os grupos (4, 5) que em sua maioria não pagou, avaliou como notas piores ou poucos avaliaram.
# + Com uma observação no grupo 5 onde os aplicativos são para maiores de idade, ou seja, um público mais restrito, onde mesmo sendo grátis os aplicativos foram bem recebidos.
# 

# In[63]:


def plotaDistClusterPlayStore():
    # plt.subplot(1 ,2 ,1)
    # plt.figure(figsize=[16, 9])
    fig, axes = plt.subplots(2, 3, figsize=[21, 16])
    k, i, j = (0, 0, 0)

    for grupo in dadosPlayStoreCluster.groupby('kmeans'):
        if(i == 2):
    #         if i==0 and j==1:
            i = 0
            j += 1
        k+=1
        grupo[1][['Rating', 'Size', 'Price', 'Reviews', 'Content Rating']].plot.box(ax=axes[i, j], title='PlayStore c '+str(k) )
        i+=1

    # fig.set_title('playStore')
        # plt.subplot(1 ,2 ,2)


    plt.show()


# # Estudo da distribuição dos agrupamentos Formados para a Play Store

# In[64]:


plotaDistClusterPlayStore()


# # Todos os grupos tiveram notas semelhantes o que reforça uma cultura de notas ao redor de 4 Play Store, talvez por serem Apps gratuitos.
# 

# In[65]:


# plt.figure(figsize=[10, 5], dpi=96)
# # sns.distplot(dadosPlayStore.Rating.dropna(), label='Google Play Store')
# # sns.distplot(dadosAppleStore.user_rating.dropna(), label='Apple Store')
# # plt.legend()

# idx = plt.subplot(221)
# plt.title('Google Play Store')
# plt.ylabel('rate')
# pd.DataFrame(dadosPlayStore.Reviews).boxplot()

# plt.subplot(222, sharey=idx)
# plt.title('Apple Store')
# pd.DataFrame(dadosAppleStore.rating_count_tot).boxplot()

# jdx = plt.subplot(223)
# plt.ylabel('kde')
# sns.distplot(dadosPlayStore.Reviews, kde=True, hist=True)

# plt.subplot(224, sharey=jdx, sharex= jdx)
# sns.distplot(dadosAppleStore.rating_count_tot, kde=True, hist=True)
# # dadosAppleStore.user_rating.min()

