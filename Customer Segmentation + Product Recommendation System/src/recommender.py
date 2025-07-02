from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_user_product_matrix(df):
    return df.pivot_table(index='customer_id', columns='product_name', values='quantity', aggfunc='sum').fillna(0)

def recommend_similar_products(user_id, user_product_matrix, top_n=5):
    cosine_sim = cosine_similarity(user_product_matrix)
    sim_df = pd.DataFrame(cosine_sim, index=user_product_matrix.index, columns=user_product_matrix.index)

    similar_users = sim_df[user_id].sort_values(ascending=False)[1:top_n+1]
    return similar_users
