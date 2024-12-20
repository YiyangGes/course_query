import re
from typing import Dict, Optional, List
from datetime import datetime

class InputValidator:
    """Input validation and sanitization for database queries"""
    
    # Restricted words that might indicate harmful queries
    RESTRICTED_WORDS = {
        'delete', 'drop', 'truncate', 'update', 'insert', 
        'alter', 'create', 'replace', 'modify', 'grant'
    }
    
    # Allowed table names from our database
    ALLOWED_TABLES = {'courses'}  # Updated to match actual database table
    
    def __init__(self):
        self.error_messages = []
    
    def validate_question(self, question: str) -> bool:
        """
        Validate a natural language question
        
        Args:
            question (str): User's input question
            
        Returns:
            bool: True if valid, False otherwise
        """
        self.error_messages = []
        
        # Check if question is empty or too long
        if not question or not question.strip():
            self.error_messages.append("Question cannot be empty")
            return False
            
        if len(question) > 500:
            self.error_messages.append("Question is too long (max 500 characters)")
            return False
        
        # Check for basic SQL injection attempts
        question_lower = question.lower()
        if any(word in question_lower for word in self.RESTRICTED_WORDS):
            self.error_messages.append("Question contains restricted keywords")
            return False
        
        # Check for excessive special characters
        if re.search(r'[;{}()\\]', question):
            self.error_messages.append("Question contains invalid characters")
            return False
            
        return True
    
    def get_error_messages(self) -> List[str]:
        """Get all error messages from validation"""
        return self.error_messages


# Add this method to the QueryValidator class:


class QueryValidator:
    """Validate generated SQL queries before execution"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.error_messages = []
    def get_error_messages(self) -> List[str]:
        """Get all error messages from validation"""
        return self.error_messages

    def validate_sql(self, sql: str) -> bool:
        """
        Validate generated SQL query
        
        Args:
            sql (str): Generated SQL query
            
        Returns:
            bool: True if valid, False otherwise
        """
        sql_lower = sql.lower()
        
        # Check if query is read-only (SELECT only)
        if not sql_lower.strip().startswith('select'):
            self.error_messages.append("Only SELECT queries are allowed")
            return False
        
        # Check for multiple statements
        # print("------"+sql.strip()[:-1]+"-------")
        if ';' in sql.strip()[:-1]:  # Allow semicolon at the end
            self.error_messages.append("Multiple SQL statements are not allowed")
            return False
        
        # Validate table names
        tables = self._extract_table_names(sql_lower)
        if not all(table in InputValidator.ALLOWED_TABLES for table in tables):
            self.error_messages.append("Query contains invalid table names")
            return False
            
        return True
    
    def _extract_table_names(self, sql: str) -> set:
        """Extract table names from SQL query"""
        # Simple regex to extract table names
        # Note: This is a basic implementation
        from_matches = re.findall(r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql)
        join_matches = re.findall(r'join\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql)
        return set(from_matches + join_matches)


class SafeQueryExecutor:
    """Safe execution of natural language queries"""
    
    def __init__(self, db_connection, query_chain, llm_chain):
        self.input_validator = InputValidator()
        self.query_validator = QueryValidator(db_connection)
        self.db = db_connection
        self.chain = llm_chain
        self.sqlgen = query_chain

    def execute_safe_query(self, question: str) -> Dict:
        """
        Safely execute a natural language query
        
        Args:
            question (str): User's natural language question
            
        Returns:
            dict: Query result and status
        """
        # Validate input
        if not self.input_validator.validate_question(question):
            return {
                'success': False,
                'error': self.input_validator.get_error_messages(),
                'query': None,
                'result': None
            }
        
        try:
            # Generate SQL query
            sql_query = self.sqlgen.invoke({"question": question})
            # print("sql from class:")
            # print(sql_query)
            # Validate generated SQL
            if not self.query_validator.validate_sql(sql_query):
                return {
                    'success': False,
                    'error': self.query_validator.get_error_messages(),
                    'query': sql_query,
                    'result': self.query_validator.get_error_messages()
                }
            
            # Execute query
            result = self.chain.invoke({"question": question})
            
            return {
                'success': True,
                'error': None,
                'query': sql_query,
                'result': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': [str(e)],
                'query': None,
                'result': None
            }

