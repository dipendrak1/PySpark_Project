import pytest
from pyspark.sql import SparkSession
from main import rename_column, add_literal_string_column, \
                read_csv_file_with_headers, filter_out_null_values


@pytest.fixture(scope="session")
def spark():
    spark = SparkSession \
        .builder \
        .master("local[*]") \
        .appName("test") \
        .getOrCreate()
    return spark


@pytest.fixture
def example_df(spark):
    data = [("Finance", 10),
            ("Marketing", 20),
            ("Sales", 30),
            ("Sales", 30),
            ("IT", 40),
            ("IT", 10)
            ]
    cols = ["dept_name", "dept_id"]

    return spark.createDataFrame(data, cols)


def test_rename_column(spark, example_df):
    renamed_df = rename_column(example_df, "dept_name", "dept_title")
    assert renamed_df.columns == ["dept_title", "dept_id"]


def test_add_literal_string_column_string(spark, example_df):
    test_df = add_literal_string_column(example_df, 'string_works', 'hello')

    data = [("Finance", 10, 'hello'),
            ("Marketing", 20, 'hello'),
            ("Sales", 30, 'hello'),
            ("Sales", 30, 'hello'),
            ("IT", 40, 'hello'),
            ("IT", 10, 'hello')
            ]
    cols = ["dept_name", "dept_id", "string_works"]

    expected_df = spark.createDataFrame(data, cols)

    assert test_df.columns == expected_df.columns
    assert dict(test_df.dtypes)["string_works"] == 'string'


def test_read_csv_file_with_headers(spark):
    test_df = read_csv_file_with_headers(spark, 'sample_file.csv')
    data = [(1, 'name_1'),
            (2, 'name_2'),
            (3, 'name_3'),
            (4, 'name_4'),
            (5, 'name_5')
            ]
    cols = ["id", "name"]
    expected_df = spark.createDataFrame(data, cols)
    assert test_df.columns == expected_df.columns
    assert (expected_df.subtract(test_df)).count() == 0


def test_filter_out_null_values(spark):
    data = [(1, 'name_1'),
            (2, 'name_2'),
            (3, 'name_3'),
            (4, 'name_4'),
            (5, None)
            ]
    example_df = spark.createDataFrame(data, ["id", "name"])

    data = [(1, 'name_1'),
            (2, 'name_2'),
            (3, 'name_3'),
            (4, 'name_4')
            ]
    expected_df = spark.createDataFrame(data, ["id", "name"])
    returned_df = filter_out_null_values(example_df, 'name')
    assert (expected_df.subtract(returned_df)).count() == 0
