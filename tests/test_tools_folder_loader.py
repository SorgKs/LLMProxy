# -*- coding: utf-8 -*-
"""
Тест динамической загрузки инструментов из папки tools/
"""
import os
import log
import tempfile
import yaml
import json
import shutil

from requests import RequestProcessor


def test_tools_from_folder_added_to_request():
    """
    Проверяет, что инструменты из папки tools/ добавляются в запрос,
    если они включены в config/tools.yaml.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tools_dir = os.path.join(tmpdir, 'tools')
        config_dir = os.path.join(tmpdir, 'config')
        os.makedirs(tools_dir)
        os.makedirs(config_dir)
        
        # Создаём тестовый инструмент в папке tools/
        test_tool = [{
            'type': 'function',
            'function': {
                'name': 'test_folder_tool',
                'description': 'A test tool from tools/ folder',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'param': {'type': 'string', 'description': 'A parameter'}
                    },
                    'required': ['param']
                }
            }
        }]
        
        with open(os.path.join(tools_dir, 'test_folder_tool.json'), 'w') as f:
            json.dump(test_tool, f)
        
        # Конфиг с включённым инструментом
        config_content = {
            'tools': {
                'test_folder_tool': True
            }
        }
        config_path = os.path.join(config_dir, 'tools.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f)
        
        processor = RequestProcessor(
            workspace_path=tmpdir,
            config_path=config_path
        )
        
        # Запрос БЕЗ tools в теле
        body = {
            'model': 'gpt-4',
            'messages': [{'role': 'user', 'content': 'test'}]
        }
        result = processor.process(body, method='POST', url='/test')
        
        # Инструмент должен быть добавлен
        assert 'tools' in result
        tool_names = [t.get('function', {}).get('name', '') for t in result['tools']]
        assert 'test_folder_tool' in tool_names, \
            f"Ожидалось наличие 'test_folder_tool' в {tool_names}"
        
        # Проверяем лог изменений
        assert any('добавлены инструменты из папки' in msg.lower() 
                  for msg in processor.changes_log), \
            "В логах должно быть сообщение об добавлении инструментов из папки"
        
        # Проверяем, что инструмент корректного формата
        added_tool = [t for t in result['tools'] 
                     if t.get('function', {}).get('name') == 'test_folder_tool'][0]
        assert added_tool['type'] == 'function'
        assert added_tool['function']['name'] == 'test_folder_tool'
        assert added_tool['function']['description'] == 'A test tool from tools/ folder'
        
        log.log_info('test_tools_from_folder_added_to_request passed!')


def test_tools_from_folder_not_added_when_disabled():
    """
    Проверяет, что инструменты НЕ добавляются, если выключены в конфиге.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tools_dir = os.path.join(tmpdir, 'tools')
        config_dir = os.path.join(tmpdir, 'config')
        os.makedirs(tools_dir)
        os.makedirs(config_dir)
        
        # Создаём тестовый инструмент
        test_tool = [{
            'type': 'function',
            'function': {
                'name': 'disabled_tool',
                'description': 'Should not be added',
                'parameters': {
                    'type': 'object',
                    'properties': {'x': {'type': 'string'}}
                }
            }
        }]
        
        with open(os.path.join(tools_dir, 'disabled_tool.json'), 'w') as f:
            json.dump(test_tool, f)
        
        # Конфиг с ВЫКЛЮЧЕННЫМ инструментом
        config_content = {
            'tools': {
                'disabled_tool': False
            }
        }
        config_path = os.path.join(config_dir, 'tools.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f)
        
        processor = RequestProcessor(
            workspace_path=tmpdir,
            config_path=config_path
        )
        
        body = {
            'model': 'gpt-4',
            'messages': [{'role': 'user', 'content': 'test'}]
        }
        result = processor.process(body)
        
        # Инструмент НЕ должен быть добавлен
        # Если tools есть, то disabled_tool не должен в нём быть
        if 'tools' in result:
            tool_names = [t.get('function', {}).get('name', '') for t in result['tools']]
            assert 'disabled_tool' not in tool_names, \
                f"'disabled_tool' не должен присутствовать в {tool_names}"
        
        log.log_info('test_tools_from_folder_not_added_when_disabled passed!')


def test_multiple_tools_from_folder():
    """
    Проверяет добавление нескольких инструментов из папки tools/.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tools_dir = os.path.join(tmpdir, 'tools')
        config_dir = os.path.join(tmpdir, 'config')
        os.makedirs(tools_dir)
        os.makedirs(config_dir)
        
        # Создаём несколько инструментов
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'tool_a',
                    'description': 'Tool A',
                    'parameters': {'type': 'object', 'properties': {}}
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'tool_b',
                    'description': 'Tool B',
                    'parameters': {'type': 'object', 'properties': {}}
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'tool_c',
                    'description': 'Tool C',
                    'parameters': {'type': 'object', 'properties': {}}
                }
            }
        ]
        
        for i, tool in enumerate(tools):
            with open(os.path.join(tools_dir, f'tool_{chr(97+i)}.json'), 'w') as f:
                json.dump([tool], f)
        
        # Конфиг со всеми включёнными
        config_content = {
            'tools': {
                'tool_a': True,
                'tool_b': True,
                'tool_c': True
            }
        }
        config_path = os.path.join(config_dir, 'tools.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f)
        
        processor = RequestProcessor(
            workspace_path=tmpdir,
            config_path=config_path
        )
        
        body = {
            'model': 'gpt-4',
            'messages': [{'role': 'user', 'content': 'test'}]
        }
        result = processor.process(body)
        
        # Все 3 инструмента должны быть добавлены
        assert 'tools' in result
        tool_names = [t.get('function', {}).get('name', '') for t in result['tools']]
        for name in ['tool_a', 'tool_b', 'tool_c']:
            assert name in tool_names, \
                f"Ожидалось {name} в {tool_names}"
        
        # Никаких других инструментов (кроме project_structure? нет, это другое поле)
        # Всего ровно 3 инструмента из папки + возможные системные (если есть)
        assert len([t for t in result['tools'] 
                   if t.get('function', {}).get('name') in ['tool_a', 'tool_b', 'tool_c']]) == 3
        
        log.log_info('test_multiple_tools_from_folder passed!')


def test_tool_already_in_request_not_duplicated():
    """
    Проверяет, что инструмент НЕ дублируется, если уже есть в запросе.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tools_dir = os.path.join(tmpdir, 'tools')
        config_dir = os.path.join(tmpdir, 'config')
        os.makedirs(tools_dir)
        os.makedirs(config_dir)
        
        # Копируем реальный function_replace.json
        shutil.copy(
            '/home/sorg/ZFS/Dev/LLMProxy/tools/function_replace.json',
            os.path.join(tools_dir, 'function_replace.json')
        )
        
        config_content = {
            'tools': {
                'function_replace': True
            }
        }
        config_path = os.path.join(config_dir, 'tools.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f)
        
        processor = RequestProcessor(
            workspace_path=tmpdir,
            config_path=config_path
        )
        
        # Запрос С УЖЕ существующим function_replace
        body = {
            'model': 'gpt-4',
            'messages': [{'role': 'user', 'content': 'test'}],
            'tools': [{
                'type': 'function',
                'function': {
                    'name': 'function_replace',
                    'description': 'Custom description',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'path': {'type': 'string'},
                            'function': {'type': 'string'},
                            'code': {'type': 'string'}
                        },
                        'required': ['path', 'function', 'code']
                    }
                }
            }]
        }
        result = processor.process(body)
        
        # Должен быть ровно ОДИН function_replace
        function_replace_tools = [
            t for t in result['tools']
            if t.get('function', {}).get('name') == 'function_replace'
        ]
        assert len(function_replace_tools) == 1, \
            f"Ожидался 1 function_replace, получено {len(function_replace_tools)}"
        
        log.log_info('test_tool_already_in_request_not_duplicated passed!')


if __name__ == '__main__':
    test_tools_from_folder_added_to_request()
    test_tools_from_folder_not_added_when_disabled()
    test_multiple_tools_from_folder()
    test_tool_already_in_request_not_duplicated()
    log.log_info('Все тесты пройдены!')
