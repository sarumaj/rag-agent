from pathlib import Path
from scrapers.ix.article import Article, ArticleEncoder, ArticleDecoder
from scrapers.ix.config import ExportFormat
import json
import pytest


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
    assert article.export_formats == []


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
        files=[Path("test.pdf")],
        export_formats=[ExportFormat(extension=".pdf", command="print", options={}, base64encoded=False)]
    )

    assert article.issue_name == "Test Issue"
    assert article.page == "42"
    assert article.title == "Test Article"
    assert article.files == [Path("test.pdf")]
    assert len(article.export_formats) == 1
    assert article.export_formats[0].extension == ".pdf"
    assert article.export_formats[0].command == "print"
    assert article.export_formats[0].options == {}
    assert article.export_formats[0].base64encoded is False


def test_article_json_serialization():
    """Test Article JSON serialization."""
    article = Article(
        id="123",
        issue_year="2024",
        issue_number="1",
        url="https://example.com/article",
        files=[Path("test.pdf")],
        export_formats=[ExportFormat(extension=".pdf", command="print", options={}, base64encoded=False)]
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
        '"files": ["test.pdf"], '
        '"export_formats": [{"extension": ".pdf", "command": "print", "options": {}, "base64encoded": false}]}'
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
        '"files": ["test.pdf"], '
        '"export_formats": [{"extension": ".pdf", "command": "print", "options": {}, "base64encoded": false}]}'
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
      "files": ["test.pdf"],
      "export_formats": [{"extension": ".pdf", "command": "print", "options": {}, "base64encoded": false}]
    }'''

    article = json.loads(json_data, cls=ArticleDecoder)
    assert isinstance(article, Article)
    assert article.id == "123"
    assert article.issue_year == "2024"
    assert article.issue_number == "1"
    assert article.url == "https://example.com/article"
    assert article.files == [Path("test.pdf")]
    assert len(article.export_formats) == 1
    assert article.export_formats[0].extension == ".pdf"
    assert article.export_formats[0].command == "print"
    assert article.export_formats[0].options == {}
    assert article.export_formats[0].base64encoded is False

    json_data = '''{
      "articles": [
        {
          "id": "123",
          "issue_year": "2024",
          "issue_number": "1",
          "url": "https://example.com/article",
          "files": ["test.pdf"],
          "export_formats": [{"extension": ".pdf", "command": "print", "options": {}, "base64encoded": false}]
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
    assert len(articles["articles"][0].export_formats) == 1
    assert articles["articles"][0].export_formats[0].extension == ".pdf"
    assert articles["articles"][0].export_formats[0].command == "print"
    assert articles["articles"][0].export_formats[0].options == {}
    assert articles["articles"][0].export_formats[0].base64encoded is False


def test_article_equality():
    """Test Article equality comparison."""
    article1 = Article(
        id="123",
        issue_year="2024",
        issue_number="1",
        url="https://example.com/article"
    )
    article2 = Article(
        id="123",
        issue_year="2024",
        issue_number="1",
        url="https://example.com/article"
    )
    article3 = Article(
        id="456",
        issue_year="2024",
        issue_number="1",
        url="https://example.com/article"
    )

    assert article1 == article2
    assert article1 != article3
    assert article2 != article3


def test_article_invalid_json_serialization():
    """Test Article JSON serialization with invalid data."""
    article = Article(
        id="123",
        issue_year="2024",
        issue_number="1",
        url="https://example.com/article",
        files=[Path("test.pdf")],
        export_formats=[ExportFormat(extension=".pdf", command="print", options={}, base64encoded=False)]
    )

    # Test with non-serializable object
    article.files.append(object())
    with pytest.raises(TypeError):
        json.dumps(article, cls=ArticleEncoder)


def test_article_json_deserialization_invalid_data():
    """Test Article JSON deserialization with invalid data."""
    # Missing required field
    json_data = '''{
      "issue_year": "2024",
      "issue_number": "1",
      "url": "https://example.com/article"
    }'''

    data = json.loads(json_data, cls=ArticleDecoder)
    assert not isinstance(data, Article)
    assert isinstance(data, dict)

    # Invalid field type
    json_data = '''{
      "id": 123,
      "issue_year": "2024",
      "issue_number": "1",
      "url": "https://example.com/article"
    }'''

    data = json.loads(json_data, cls=ArticleDecoder)
    assert not isinstance(data, Article)
    assert isinstance(data, dict)


def test_article_json_deserialization_extra_fields():
    """Test Article JSON deserialization with extra fields."""
    json_data = '''{
      "id": "123",
      "issue_year": "2024",
      "issue_number": "1",
      "url": "https://example.com/article",
      "extra_field": "value"
    }'''

    data = json.loads(json_data, cls=ArticleDecoder)
    assert not isinstance(data, Article)
    assert isinstance(data, dict)


def test_article_json_deserialization_empty_list():
    """Test Article JSON deserialization with empty lists."""
    json_data = '''{
      "id": "123",
      "issue_year": "2024",
      "issue_number": "1",
      "url": "https://example.com/article",
      "files": [],
      "export_formats": []
    }'''

    article = json.loads(json_data, cls=ArticleDecoder)
    assert isinstance(article, Article)
    assert article.files == []
    assert article.export_formats == []


def test_article_json_deserialization_nested_objects():
    """Test Article JSON deserialization with nested objects."""
    json_data = '''{
      "id": "123",
      "issue_year": "2024",
      "issue_number": "1",
      "url": "https://example.com/article",
      "files": ["test1.pdf", "test2.pdf"],
      "export_formats": [
        {"extension": ".pdf", "command": "print", "options": {}, "base64encoded": false},
        {"extension": ".mhtml", "command": "save", "options": {"format": "mhtml"}, "base64encoded": false}
      ]
    }'''

    article = json.loads(json_data, cls=ArticleDecoder)
    assert isinstance(article, Article)
    assert len(article.files) == 2
    assert len(article.export_formats) == 2
    assert article.export_formats[0].extension == ".pdf"
    assert article.export_formats[1].extension == ".mhtml"
