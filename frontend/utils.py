import pandas as pd
def movie_type_to_dict():
    l=[]
    type_list=['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
    for idx in range(0,len(type_list)):
        type_dict = {}

        type_dict['id']=idx
        type_dict['movie_type']=type_list[idx]
        l.append(type_dict)
    return l



def get_movie_list_by_year_type(year, genre):
    movies_title = ['MovieID', 'Title', 'Genres']
    print("here, generating movies by type ---------------", genre)
    movies = pd.read_table('movielens/ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python', encoding='ISO-8859-1')

    # Split the Title column and create separate columns for Title and Year
    movies[['Title', 'Year']] = movies['Title'].str.split('(', n=1, expand=True)
    movies['Year'] = movies['Year'].str.replace(')', '').str.strip()
    movies['Year'] = movies['Year'].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, ignore errors

    movies['Genres'] = movies['Genres'].str.split('|')

    result = []
    for index, row in movies.iterrows():
        if pd.notna(row['Year']) and year == int(row['Year']) and genre in row['Genres']:
            result.append(row.tolist())

    print("vvvvvvvvvv", result)
    return result


# TEST CASE
# year = 2000
# genre = "Drama"
# result = get_movie_list_by_year_type(year, genre)
# print(result)