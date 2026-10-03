原則
- 命名規則請見 naming.md。
- 參數與回傳值都要寫 type hint。
- 有回傳值的函數用 return 結尾。
- 每行不超過 80 個字元。

```python
# 函數名稱可以多個單字組合，中間用 "_" 符號分開。
def function_sample(variable_name: str) -> bool:
    # variable_name: 參數名稱採用多個單字組合，中間用 "_" 符號分開。
    
    return True
```