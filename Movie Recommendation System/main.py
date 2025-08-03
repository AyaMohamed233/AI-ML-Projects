import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_columns)

movies_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL']
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', names=movies_columns, usecols=range(5))

data = pd.merge(ratings, movies, on='movie_id')

def build_user_item_matrix(df):
    return df.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

def get_user_similarity(user_item_matrix):
    similarity = cosine_similarity(user_item_matrix)
    return pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_movies(target_user_id, data, user_similarity_df, top_n=5):
    similar_users = user_similarity_df.loc[target_user_id].sort_values(ascending=False)[1:top_n+1]
    similar_users_ids = similar_users.index
    watched = data[data['user_id'] == target_user_id]['title'].tolist()
    similar_users_ratings = data[data['user_id'].isin(similar_users_ids)]
    recommendations = similar_users_ratings[~similar_users_ratings['title'].isin(watched)]
    movie_scores = recommendations.groupby('title')['rating'].mean()
    return movie_scores.sort_values(ascending=False)

def precision_at_k(recommended_titles, test_liked, k=10):
    top_k = recommended_titles[:k]
    relevant = [title for title in top_k if title in test_liked]
    return len(relevant) / k

target_user_id = 3
user_data = data[data['user_id'] == target_user_id]
train_data, test_data = train_test_split(user_data, test_size=0.3, random_state=42)
train_movies = train_data['title'].tolist()
training_data = data[data['user_id'] != target_user_id]
training_data = pd.concat([training_data, train_data])

user_item_matrix = build_user_item_matrix(training_data)
user_similarity_df = get_user_similarity(user_item_matrix)
movie_scores = recommend_movies(target_user_id, training_data, user_similarity_df)
top_k_recommendations = movie_scores.head(10)

print("\nTop 10 Recommended Movies (User-Based):")
print(top_k_recommendations)

recommended_titles = top_k_recommendations.index.tolist()
test_liked = test_data[test_data['rating'] >= 4]['title'].tolist()
score = precision_at_k(recommended_titles, test_liked, k=10)

print(f"\nPrecision@10 = {score:.2f}")

item_user_matrix = data.pivot_table(index='title', columns='user_id', values='rating').fillna(0)
item_similarity = cosine_similarity(item_user_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)

target_movie = 'Star Wars (1977)'
similar_items = item_similarity_df[target_movie].sort_values(ascending=False)[1:6]

print("\nTop 5 Similar Movies to 'Star Wars (1977)':")
print(similar_items)
