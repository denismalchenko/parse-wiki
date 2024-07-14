"""
Модуль хранения графа ссылок статей википедии
"""
import logging
import requests
import concurrent.futures
import os
import json
import dotenv
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup, Tag
from colorlog import ColoredFormatter
from enum import Enum, auto
from pynput import keyboard
from neo4j import GraphDatabase
from itertools import groupby

class StopParsing(Enum):
    LEVEL = auto()
    ARTICLE = auto()
    

class WikiGraph:
    """Класс для хранения графа ссылок статей википедии""" 

    def __init__(self, title: str) -> None:
        self._data: list[dict[str, list[str]]] = list() # Граф статей и ссылок
        self._titles: set[str] = set() # Все спарсенные статьи
        self._levels: int = 0 # Уровень графа
        self._current_level_unparsed: set[str] = set([title]) # Нераспарсенные ссылки на текущем уровне
        self._next_level_unparsed: set[str] = set() # Нераспарсенные ссылки на следующем уровне
        self.bad_titles: set[str] = set()
        self.parcer = WikiGraphParser()

    def __len__(self) -> int:
        return len(self._titles)
     
    def __bool__(self) -> bool:
        return bool(self._current_level_unparsed)
    
    def get_data(self) -> list[dict[str, list[str]]]:
        return self._data

    def get_unparsed(self) -> set[str]:
        return self._current_level_unparsed

    def get_level(self) -> int:
        return self._levels

    def incr_level(self) -> None:
        self._levels += 1

    def get_titles(self) -> set[str]:
        return self._titles

    def is_parsed(self, title: str) -> bool:
        return title in self._titles
    
    def parse(self, stoptype: StopParsing, maxvalue: int) -> None:
        self.parcer.parse(self, stoptype, maxvalue)

    def unparsed_update(self) -> None:
        self._current_level_unparsed -= self._titles | self.bad_titles
        if self._titles and not self._current_level_unparsed:
            self._current_level_unparsed, self._next_level_unparsed, self.bad_titles = self._next_level_unparsed, set(), set()
            self.incr_level()

    def add_bad_title(self, title: str) -> None:
        self.bad_titles.add(title)

    def append(self, item: dict[str, list[str]]) -> None:
        self._titles.add(item['title'])
        self._data.append(item)
        for ref in item['references']:
            self._next_level_unparsed.add(ref)

    def back(self) -> None:
        self._titles.discard(self._data[-1]['title'])
        self._data.pop()

    @staticmethod
    def get_wiki_url(title: str) -> str:
        return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

class WikiGraphParser:
    """Класс для парсинга статей википедии"""
    # Эти идентификаторы будут пропускаться при парсинге, т.к. не содержат внутренние ссылки
    BAD_IDS: list[str] = [
        'References', 'catlinks', 'External_links', 'Notes', 'Footnotes',
        'Citations'
    ]
    # Эти строки будут пропускаться при парсинге, т.к. содержат не верные внутренние ссылки
    BAD_STRINGS: list[str] = [
        'cite_note', 'Citation_needed', 'NOTRS', 'https', '(disambiguation)',
        ':', 'action='
    ]

    def __init__(self) -> None:
        self.thread_number: int = 6
        self.logger = self.init_logger()

    def init_logger(self) -> logging.Logger:
        logger = logging.getLogger(f'wiki parser-{id(self)}')
        logger.setLevel(logging.INFO)
        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)s%(reset)s:%(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def parse_article(self, wiki_graph: WikiGraph, title: str,) -> bool:
        if wiki_graph.is_parsed(title):
            return False  # Статья уже была обработана
        html_content = BeautifulSoup(
            requests.get(wiki_graph.get_wiki_url(title), timeout=10).content, "html.parser")
        if html_content.select_one('.noarticletext'):
            wiki_graph.add_bad_title(title)
            return False  # Статья не существует или пустая
        article: dict[str, set[str]] = {'title': title, 'references': set()}
        for element in html_content.select_one('.mw-content-ltr').next_elements:
            if isinstance(element,
                          Tag) and element.name == 'a' and element.get('href'):
                if element.get('id') in self.BAD_IDS:
                    break
                href = element.get('href')
                if '/wiki' in href and not any(
                        bad_string in href for bad_string in self.BAD_STRINGS):
                    article['references'].add(element.get('title'))
        wiki_graph.append({'title': title, 'references': list(article['references'])})            
        return True
    
    def on_key_press(self, key: keyboard.Key) -> None:
        if key == keyboard.Key.esc:
            self.exit_flag = True
            print(self.exit_flag)

    def thread_parse(self, wiki_graph: WikiGraph, title: str, maxvalue: int|None = None) -> None:
        if self.parse_article(wiki_graph, title):
            if maxvalue and len(wiki_graph) > maxvalue:
                wiki_graph.back()
            else:
                self.logger.info(f'{len(wiki_graph): 9d}: Completed parsing of the article "{title}"')

    def parse(self, wiki_graph: WikiGraph, stoptype: StopParsing, maxvalue: int) -> None:
        wiki_graph.unparsed_update()
        if (stoptype == StopParsing.ARTICLE and len(wiki_graph) >= maxvalue) or (stoptype == StopParsing.LEVEL and wiki_graph.get_level() >= maxvalue):
            self.logger.info('Current graph already contains enough level or articles. Stop parsing.')
            return
        print("You can press 'esc' to stop the parsing.")
        self.exit_flag = False
        with keyboard.Listener(on_press=lambda key: self.on_key_press(key)) as listener:
            back_flag: int | None = maxvalue if stoptype == StopParsing.ARTICLE else None
            while not self.exit_flag:
                self.logger.info('Parsing level %d', wiki_graph.get_level()+1)
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_number) as executor:
                    futures = [executor.submit(self.thread_parse, wiki_graph, title, back_flag) for title in wiki_graph.get_unparsed()]
                    for future in concurrent.futures.as_completed(futures):
                        if self.exit_flag or (stoptype == StopParsing.ARTICLE and len(wiki_graph) >= maxvalue):
                            self.exit_flag = True
                            break
                    executor.shutdown(cancel_futures=True)
                wiki_graph.unparsed_update()
                if (stoptype == StopParsing.LEVEL and wiki_graph.get_level() >= maxvalue) or len(wiki_graph) == 0 or not wiki_graph:
                    self.exit_flag = True
        if len(wiki_graph) == 0:
            self.logger.info('This title does not exist in the wiki.')
        elif stoptype == StopParsing.LEVEL and wiki_graph.get_level() >= maxvalue:
            self.logger.info('Graph %d level is completely parsed.', maxvalue)
        elif stoptype == StopParsing.ARTICLE and len(wiki_graph) >= maxvalue:
            self.logger.info('%d articles is completely parsed.', len(wiki_graph))
        elif not wiki_graph:
            self.logger.info('Graph is completely parsed.')
        else:
            self.logger.info('You stopped the parsing process.')


class StrategyIO(ABC):
    @abstractmethod
    def save(self, wiki_graph: WikiGraph, filename: str) -> None:
        pass

    @abstractmethod
    def load(self, wiki_graph: WikiGraph, filename: str) -> None:
        pass

class JsonIO(StrategyIO):
    def save(self, wiki_graph: WikiGraph, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(wiki_graph.get_data(), f, ensure_ascii=False, indent=4)

    def load(self, wiki_graph: WikiGraph, filename: str) -> list[dict[str, list[str]]]:
        with open(filename, 'r', encoding='utf-8') as f:
            data = wiki_graph.append(json.load(f))
        return data
    
class Neo4jIO(StrategyIO):

    def is_not_env_file(self, filename):
        _, extension = os.path.splitext(filename)
        return extension.lower() != '.env'
    
    def get_driver(self, filename: str) -> GraphDatabase.driver:
        if self.is_not_env_file(filename):
            raise ValueError('The file must be ".env"')
        dotenv.load_dotenv(filename)
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')
        return GraphDatabase.driver(uri, auth=(user, password))
    
    def save(self, wiki_graph: WikiGraph, filename: str) -> None:
        driver = self.get_driver(filename)
        data = wiki_graph.get_data()
        with driver.session() as session:
            session.run("MERGE (t:Title {name: $title})",
                        title=data[0]['title'])
            for titles in data:
                ref_list = titles['references']
                chunk_size = 30
                for i in range(0, len(ref_list), chunk_size):
                    chunk = ref_list[i:i + chunk_size]
                    merge_statements = " ".join([
                        f"MERGE (r{j}:Title {{name: $ref_title{j}}})"
                        for j in range(len(chunk))
                    ])
                    create_statements = " ".join([
                        f"CREATE (t)-[:REFERS_TO]->(r{k})"
                        for k in range(len(chunk))
                    ])
                    query = ("MATCH (t:Title {name: $title}) " +
                             merge_statements + create_statements)
                    parameters = {"title": titles['title']}
                    parameters.update({
                        f"ref_title{l}": ref_list[i + l]
                        for l in range(len(chunk))
                    })
                    session.run(query, parameters)
        driver.close()

    def load(self, wiki_graph: WikiGraph, filename: str) -> list[dict[str, list[str]]]:
        driver = self.get_driver(filename)
        raw_data = []
        with driver.session() as session:
            for record in session.run("MATCH (t:Title)-[:REFERS_TO]->(r:Title) RETURN t.name AS t1, r.name AS t2"):
                raw_data.append((record.t1, record.t2))
        driver.close()
        raw_data.sort(key=lambda x: x[0])
        grouped_data = []
        for key, group in groupby(raw_data, lambda x: x[0]):
            grouped_data.append((key, list(group)))
        data = []
        for title, tuple_titles in grouped_data:
            data.append({'title': title, 'references': [t[1] for t in tuple_titles]})
        return data
