from pathlib import Path
from scrappers.ix.article import Article, ArticleEncoder, ArticleDecoder
import json


def test_article_creation():
    """Test basic Article creation with required fields."""
    article = Article(
        id="123",
        issue_year="2024",
        issue_number="1",
        url="https://example.com/article"
    )

    assert article.id == "123"
    assert article.issue_year == "2024"
    assert article.issue_number == "1"
    assert article.url == "https://example.com/article"
    assert article.issue_name == ""
    assert article.page == ""
    assert article.title == ""
    assert article.files == []


def test_article_with_optional_fields():
    """Test Article creation with optional fields."""
    article = Article(
        id="123",
        issue_year="2024",
        issue_number="1",
        url="https://example.com/article",
        issue_name="Test Issue",
        page="42",
        title="Test Article",
        files=[Path("test.pdf")]
    )

    assert article.issue_name == "Test Issue"
    assert article.page == "42"
    assert article.title == "Test Article"
    assert article.files == [Path("test.pdf")]


def test_article_json_serialization():
    """Test Article JSON serialization."""
    article = Article(
        id="123",
        issue_year="2024",
        issue_number="1",
        url="https://example.com/article",
        files=[Path("test.pdf")]
    )

    json_str = json.dumps(article, cls=ArticleEncoder)
    assert json_str == (
        '{"id": "123", '
        '"issue_year": "2024", '
        '"issue_number": "1", '
        '"url": "https://example.com/article", '
        '"issue_name": "", '
        '"page": "", '
        '"title": "", '
        '"files": ["test.pdf"]}'
    )

    json_str = json.dumps({"articles": [article]}, cls=ArticleEncoder)
    assert json_str == (
        '{"articles": [{"id": "123", '
        '"issue_year": "2024", '
        '"issue_number": "1", '
        '"url": "https://example.com/article", '
        '"issue_name": "", '
        '"page": "", '
        '"title": "", '
        '"files": ["test.pdf"]}'
        ']}'
    )


def test_article_json_deserialization():
    """Test Article JSON deserialization."""
    json_data = '''{
      "id": "123",
      "issue_year": "2024",
      "issue_number": "1",
      "issue_name": "",
      "page": "",
      "title": "",
      "url": "https://example.com/article",
      "files": ["test.pdf"]
    }'''

    article = json.loads(json_data, cls=ArticleDecoder)
    assert isinstance(article, Article)
    assert article.id == "123"
    assert article.issue_year == "2024"
    assert article.issue_number == "1"
    assert article.url == "https://example.com/article"
    assert article.files == [Path("test.pdf")]

    json_data = '''{
      "articles": [
        {
          "id": "123",
          "issue_year": "2024",
          "issue_number": "1",
          "url": "https://example.com/article",
          "files": ["test.pdf"]
        }
      ]
    }'''

    articles = json.loads(json_data, cls=ArticleDecoder)
    assert isinstance(articles["articles"], list)
    assert len(articles["articles"]) == 1
    assert isinstance(articles["articles"][0], Article)
    assert articles["articles"][0].id == "123"
    assert articles["articles"][0].issue_year == "2024"
    assert articles["articles"][0].issue_number == "1"
    assert articles["articles"][0].url == "https://example.com/article"
    assert articles["articles"][0].files == [Path("test.pdf")]
