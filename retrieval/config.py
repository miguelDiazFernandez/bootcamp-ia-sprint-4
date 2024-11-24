from __future__ import annotations

from typing import Callable

from data_utils import Movie
from retrieval.indexing_pipeline_utils import get_synopsys_txt
from retrieval.retrieval_pipeline_utils import clean_query_txt


class RetrievalExpsConfig:
    """
    Class to keep track of all the parameters used in the embeddings experiments.
    Any attribute created in this class will be logged to mlflow.

    Nota: cuando definimos atributos de tipo Callable, debemos usar `staticmethod` para que la función pueda ser llamada
    s
    """

    def __init__(
        self,
        text_to_embed_fn: Callable[[Movie], str] = get_synopsys_txt,
        query_prepro_fn: Callable[[str], str] = clean_query_txt,
        model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = False,
    ):
        # Configuración inicial
        self._text_to_embed_fn: Callable[[Movie], str] = text_to_embed_fn
        self._query_prepro_fn: Callable[[str], str] = query_prepro_fn
        self.model_name: str = model_name
        self.normalize_embeddings: bool = normalize_embeddings

        # Validación de funciones y parámetros
        self._validate_functions()

    def _validate_functions(self):
        """
        Valida que las funciones y configuraciones sean correctas.
        Lanza un ValueError si alguna validación falla.
        """
        if not callable(self._text_to_embed_fn):
            raise ValueError("El atributo _text_to_embed_fn debe ser una función válida que reciba un objeto `Movie` y devuelva un string.")
        if not callable(self._query_prepro_fn):
            raise ValueError("El atributo _query_prepro_fn debe ser una función válida que reciba un string y devuelva un string.")
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("El atributo model_name debe ser una cadena no vacía.")
        if not isinstance(self.normalize_embeddings, bool):
            raise ValueError("El atributo normalize_embeddings debe ser un valor booleano.")

    ## NO MODIFICAR A PARTIR DE AQUÍ ##
        """
        Valida que las funciones y configuraciones sean correctas.
        Lanza un ValueError si alguna validación falla.
        """
        if not callable(self._text_to_embed_fn):
            raise ValueError("El atributo _text_to_embed_fn debe ser una función válida que reciba un objeto `Movie` y devuelva un string.")
        if not callable(self._query_prepro_fn):
            raise ValueError("El atributo _query_prepro_fn debe ser una función válida que reciba un string y devuelva un string.")
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("El atributo model_name debe ser una cadena no vacía.")
        if not isinstance(self.normalize_embeddings, bool):
            raise ValueError("El atributo normalize_embeddings debe ser un valor booleano.")

    ## NO MODIFICAR A PARTIR DE AQUÍ ##

    def text_to_embed_fn(self, movie: Movie) -> str:
        return self._text_to_embed_fn(movie)

    def query_prepro_fn(self, query: dict) -> str:
        return self._query_prepro_fn(query)

    @property
    def index_config_unique_id(self) -> str:
        mname = self.model_name.replace("/", "_")
        return f"{mname}_{self._text_to_embed_fn.__name__}_{self.normalize_embeddings}"

    @property
    def exp_params(self) -> dict:
        """
        Return the config parameters as a dictionary. To be used, for example, in mlflow logging
        """
        return {
            "model_name": self.model_name,
            "text_to_embed_fn": self._text_to_embed_fn.__name__,
            "normalize_embeddings": self.normalize_embeddings,
            "query_prepro_fn": self._query_prepro_fn.__name__,
        }
