from __future__ import annotations

from langchain_core.documents import Document

from data_utils import Movie
from retrieval import config


def create_docs_to_embedd(movies: list[Movie], config: config.RetrievalExpsConfig) -> list[Document]:
    """
    Convierte una lista de objetos `Movie` a una lista the objetos `Document`(usada por Langchain).
    En esta función se decide que parte de los datos será usado como embeddings y que parte como metadata.
    """
    movies_as_docs = []
    for movie in movies:
        content = config.text_to_embed_fn(movie)
        metadata = movie.model_dump()
        doc = Document(page_content=content, metadata=metadata)
        movies_as_docs.append(doc)

    return movies_as_docs


## Posibles funciones para usar como `text_to_embed_fn` en `RetrievalExpsConfig` ##


def get_synopsys_txt(movie: Movie) -> str:
    return movie.synopsis or "Sinopsis no disponible"

def get_title_and_synopsys_txt(movie: Movie) -> str:

    title = movie.title_original or "Título desconocido"
    synopsis = movie.synopsis or "Sinopsis no disponible"
    return f"{title}: {synopsis}"

def get_title_genre_synopsis_txt(movie: Movie) -> str:

    title = movie.title_original or "Título desconocido"
    genres = (movie.genre_tags or "Géneros no disponibles").replace(";", ", ")
    synopsis = movie.synopsis or "Sinopsis no disponible"
    return f"Título: {title}\nGéneros: {genres}\nSinopsis: {synopsis}"

def get_extended_info_txt(movie: Movie) -> str:

    title = movie.title_original or "Título desconocido"
    genres = (movie.genre_tags or "Géneros no disponibles").replace(";", ", ")
    directors = movie.director_top_5 or "Director no disponible"
    synopsis = movie.synopsis or "Sinopsis no disponible"
    return (f"Título: {title}\n"
            f"Géneros: {genres}\n"
            f"Director: {directors}\n"
            f"Sinopsis: {synopsis}")

def get_minimal_info_txt(movie: Movie) -> str:

    title = movie.title_original or "Título desconocido"
    genres = (movie.genre_tags or "Géneros no disponibles").replace(";", ", ")
    return f"Título: {title}\nGéneros: {genres}"
