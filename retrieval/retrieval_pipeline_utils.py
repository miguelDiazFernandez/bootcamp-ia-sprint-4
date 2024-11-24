def clean_query_txt(query: str) -> str:

    if not query:
        return "Consulta no especificada"

    query = query.replace("El usuario busca ", "").strip()
    query = " ".join(query.split())
    query = query.lower()
    return query
