# SQLite 从零到企业级 完全学习指南

> 目标：从"数据库是什么"到"能在生产项目中正确使用 SQLite"

---

## 目录

1. [SQLite 是什么 — 概念篇](#1-sqlite-是什么--概念篇)
2. [5 分钟快速上手](#2-5-分钟快速上手)
3. [数据类型详解](#3-数据类型详解)
4. [CRUD 增删改查 — 基础 SQL 语句](#4-crud-增删改查--基础-sql-语句)
5. [查询进阶 — WHERE / JOIN / GROUP BY / 子查询](#5-查询进阶--where--join--group-by--子查询)
6. [内置函数大全](#6-内置函数大全)
7. [表管理 — 约束 / 修改 / 删除](#7-表管理--约束--修改--删除)
8. [索引与性能优化](#8-索引与性能优化)
9. [事务与并发 — ACID / WAL / 锁](#9-事务与并发--acid--wal--锁)
10. [Python sqlite3 模块详解](#10-python-sqlite3-模块详解)
11. [进阶特性 — CTE / 窗口函数 / 触发器 / 视图](#11-进阶特性--cte--窗口函数--触发器--视图)
12. [生产环境最佳实践](#12-生产环境最佳实践)
13. [常见错误与反模式](#13-常见错误与反模式)
14. [企业级封装示例](#14-企业级封装示例)

---

## 1. SQLite 是什么 — 概念篇

### 一句话理解

SQLite 是一个**不需要安装服务器**的数据库。整个数据库就是一个 `.db` 文件，你直接读写它。

### 类比理解

| 概念 | 类比 |
|------|------|
| SQLite | 一个 Excel 文件（自包含、复制即用） |
| MySQL / PostgreSQL | 一台专门的"数据服务器"（需要安装、启动、连接） |
| SQLite 的 `.db` 文件 | 就像 `.xlsx`：双击打不开，但程序可以读写 |
| `sqlite3` 命令行 | 就像用 Excel 打开 `.xlsx` 查看内容 |

### SQLite vs 其他数据库

| 对比维度 | SQLite | MySQL / PostgreSQL |
|---------|--------|-------------------|
| 架构 | **嵌入式** — 读写 `.db` 文件 | **客户端-服务器** — 连接数据库服务 |
| 安装 | Python 内置 `import sqlite3` | 需要安装服务 + 配置端口/用户/密码 |
| 部署 | 零部署，文件在哪库就在哪 | 需要运维数据库服务器 |
| 并发写入 | **单写**（同一时间只能一个进程写） | 高并发读写，支持连接池 |
| 适用规模 | 单机、App、原型、小型 Web（日活 < 10万） | 多用户、高并发、大规模数据 |
| 速度（读取） | 极快（本地文件读取） | 需要网络传输 |
| 配置 | 几乎零配置 | 需要调优参数 |
| 备份 | 复制 `.db` 文件就行 | 需要 `mysqldump` / `pg_dump` |

### 什么场景用 SQLite

- ✅ 本地应用（手机 App、桌面软件、游戏存档）
- ✅ 学习和原型开发
- ✅ 单机工具、CLI 工具
- ✅ AI Agent 的记忆存储（你的 step13 就在用）
- ✅ 小型 Web 应用（搭配 WAL 模式）
- ✅ 数据分析（临时处理 CSV 导入 SQLite 用 SQL 分析）

### 什么场景不适合 SQLite

- ❌ 高并发写（每秒百次以上写入、多个进程同时写）
- ❌ 超大规模（TB 级数据）
- ❌ 分布式系统（需要多台机器共享数据库）

---

## 2. 5 分钟快速上手

### Python 方式（零依赖）

```python
import sqlite3

# 1. 连接数据库（文件不存在则自动创建）
conn = sqlite3.connect("demo.db")

# 2. 创建游标（用来执行 SQL 语句的东西）
cur = conn.cursor()

# 3. 创建表
cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        email TEXT UNIQUE
    )
""")

# 4. 插入数据
cur.execute(
    "INSERT INTO users (name, age, email) VALUES (?, ?, ?)",
    ("张三", 28, "zhangsan@example.com")
)
conn.commit()  # 提交事务（不提交不会保存！）

# 5. 查询数据
cur.execute("SELECT * FROM users")
rows = cur.fetchall()
for row in rows:
    print(dict(row))  # {'id': 1, 'name': '张三', 'age': 28, ...}

# 6. 关闭连接
conn.close()
```

### 命令行方式（安装 sqlite3 工具）

```bash
# 进入 SQLite 命令行
sqlite3 demo.db

# 查看所有表
.tables

# 查看表结构
.schema users

# 查询
SELECT * FROM users;

# 退出
.quit
```

---

## 3. 数据类型详解

### SQLite 的 5 种存储类型

SQLite 使用**动态类型系统**（不像 MySQL 严格区分 INT/VARCHAR）。

| 存储类型 | 说明 | Python 对应 |
|---------|------|------------|
| `NULL` | 空值 | `None` |
| `INTEGER` | 整数（1, 2, 3, 4, 6, 8 字节自动选择） | `int` |
| `REAL` | 浮点数（8 字节 IEEE） | `float` |
| `TEXT` | 文本（UTF-8 / UTF-16） | `str` |
| `BLOB` | 二进制数据（原样存储） | `bytes` |

### 常用字段定义

```sql
-- 自增主键（最常用的 ID 字段）
id INTEGER PRIMARY KEY AUTOINCREMENT

-- 文本（必填）
name TEXT NOT NULL

-- 整数（默认值）
age INTEGER DEFAULT 0

-- 浮点数
price REAL NOT NULL

-- 布尔值（SQLite 没有 BOOL，用 INTEGER 0/1 替代）
is_active INTEGER DEFAULT 1

-- 时间（用 TEXT 存 ISO 格式，最通用）
created_at TEXT DEFAULT (datetime('now'))

-- 唯一约束（不能重复）
email TEXT UNIQUE

-- 外键
user_id INTEGER REFERENCES users(id)
```

### 类型亲和力（进阶）

SQLite 的 `VARCHAR(255)` 不是真正限制 255 个字符，它可以存任意长度。类型亲和力规则：

```sql
-- 这些声明都会映射到 INTEGER 亲和力
INT, INTEGER, TINYINT, SMALLINT, MEDIUMINT, BIGINT

-- 这些映射到 TEXT 亲和力
TEXT, CLOB, CHAR, VARCHAR, CHARACTER

-- 这些映射到 REAL
REAL, DOUBLE, FLOAT

-- 这些映射到 NONE（不转换）
BLOB, 不指定类型
```

---

## 4. CRUD 增删改查 — 基础 SQL 语句

CRUD = Create（增）/ Read（查）/ Update（改）/ Delete（删）

### 4.1 INSERT — 插入数据

```sql
-- 基础插入：指定列名（推荐，列顺序无关）
INSERT INTO users (name, age, email) VALUES ('张三', 28, 'zhang@example.com');

-- 简写：省略列名时 VALUES 必须和建表顺序一致（不推荐）
INSERT INTO users VALUES (NULL, '李四', 30, 'li@example.com');

-- 批量插入（性能比多次单条插入快 10 倍以上）
INSERT INTO users (name, age, email) VALUES
    ('王五', 25, 'wang@example.com'),
    ('赵六', 32, 'zhao@example.com'),
    ('孙七', 27, 'sun@example.com');

-- 冲突处理：如果 email 已存在则更新 name
INSERT INTO users (name, age, email) VALUES ('张三改', 29, 'zhang@example.com')
ON CONFLICT(email) DO UPDATE SET name = excluded.name, age = excluded.age;

-- 冲突处理：如果已存在则忽略（不报错）
INSERT OR IGNORE INTO users (name, email) VALUES ('张三', 'zhang@example.com');

-- 插入查询结果
INSERT INTO archive_users SELECT * FROM users WHERE age > 30;
```

### 4.2 SELECT — 查询数据

```sql
-- 查询所有列（* 表示所有列）
SELECT * FROM users;

-- 查询指定列
SELECT name, email FROM users;

-- 去重
SELECT DISTINCT age FROM users;

-- 别名（给列起临时名字）
SELECT name AS 姓名, age AS 年龄 FROM users;

-- 限制返回行数
SELECT * FROM users LIMIT 10;

-- 跳过前 N 行后取 M 行（分页查询：第 2 页，每页 10 条）
SELECT * FROM users LIMIT 10 OFFSET 10;

-- 排序（ASC 升序 / DESC 降序）
SELECT * FROM users ORDER BY age DESC;
SELECT * FROM users ORDER BY age ASC, name DESC;  -- 多列排序

-- 条件查询（下一章详解）
SELECT * FROM users WHERE age > 25 AND age < 35;
```

### 4.3 UPDATE — 更新数据

```sql
-- 基础更新（⚠️ 没有 WHERE 会更新所有行！）
UPDATE users SET age = 29 WHERE name = '张三';

-- 更新多列
UPDATE users SET age = 30, email = 'new@example.com' WHERE id = 1;

-- 基于表达式更新
UPDATE users SET age = age + 1;  -- 所有人年龄 +1

-- 用子查询更新
UPDATE users SET age = (SELECT AVG(age) FROM users) WHERE id = 1;
```

### 4.4 DELETE — 删除数据

```sql
-- 基础删除（⚠️ 没有 WHERE 会删除所有行！）
DELETE FROM users WHERE id = 5;

-- 删除多条件
DELETE FROM users WHERE age < 18 OR age > 65;

-- 清空整张表（比 DELETE 快，且重置自增 ID）
DELETE FROM users;  -- 逐行删除，触发 trigger
-- 或者
TRUNCATE TABLE users;  -- SQLite 不支持，用 DELETE FROM 替代
```

### 4.5 Python 中的参数化查询（防 SQL 注入）

```python
# ✅ 正确：用 ? 占位符（防 SQL 注入）
cur.execute("SELECT * FROM users WHERE name = ?", ("张三",))

# ✅ 正确：多个参数
cur.execute(
    "INSERT INTO users (name, age) VALUES (?, ?)",
    ("张三", 28)
)

# ✅ 正确：使用字典参数
cur.execute(
    "INSERT INTO users (name, age) VALUES (:name, :age)",
    {"name": "张三", "age": 28}
)

# ✅ 批量执行
data = [("张三", 28), ("李四", 30), ("王五", 25)]
cur.executemany("INSERT INTO users (name, age) VALUES (?, ?)", data)

# ❌ 错误：拼接字符串（SQL 注入风险！）
name = "张三'; DROP TABLE users; --"
cur.execute(f"SELECT * FROM users WHERE name = '{name}'")  # 危险！
```

---

## 5. 查询进阶 — WHERE / JOIN / GROUP BY / 子查询

### 5.1 WHERE 条件运算符

```sql
-- 比较运算符：=  !=  >  <  >=  <=
SELECT * FROM users WHERE age >= 18;

-- 逻辑运算符：AND / OR / NOT
SELECT * FROM users WHERE age >= 18 AND age <= 60;
SELECT * FROM users WHERE city = '北京' OR city = '上海';

-- BETWEEN（包含边界）
SELECT * FROM users WHERE age BETWEEN 18 AND 60;

-- IN（匹配列表中的值）
SELECT * FROM users WHERE city IN ('北京', '上海', '杭州');

-- NOT IN
SELECT * FROM users WHERE city NOT IN ('深圳', '广州');

-- LIKE 模糊匹配（% 匹配任意字符，_ 匹配单个字符）
SELECT * FROM users WHERE name LIKE '张%';    -- 姓张的
SELECT * FROM users WHERE email LIKE '%@gmail.com';  -- Gmail 邮箱
SELECT * FROM users WHERE phone LIKE '138________';  -- 138 开头的 11 位手机号

-- IS NULL / IS NOT NULL
SELECT * FROM users WHERE email IS NULL;
SELECT * FROM users WHERE email IS NOT NULL;

-- EXISTS（子查询返回结果则为 True）
SELECT * FROM users WHERE EXISTS (
    SELECT 1 FROM orders WHERE orders.user_id = users.id
);
```

### 5.2 聚合查询 — GROUP BY

```sql
-- COUNT / AVG / SUM / MAX / MIN
SELECT COUNT(*) FROM users;                    -- 总行数
SELECT AVG(age) FROM users;                    -- 平均年龄
SELECT SUM(salary) FROM users;                 -- 工资总和
SELECT MAX(age), MIN(age) FROM users;          -- 最大最小年龄

-- 分组统计
SELECT city, COUNT(*) AS 人数 FROM users
GROUP BY city;

-- 分组 + 过滤（HAVING 过滤分组后的结果，WHERE 过滤分组前的行）
SELECT city, AVG(age) AS 平均年龄 FROM users
GROUP BY city
HAVING AVG(age) > 30;  -- 只显示平均年龄 > 30 的城市

-- 完整执行顺序：FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
SELECT city, COUNT(*) AS n
FROM users
WHERE age >= 18           -- 先筛选
GROUP BY city             -- 再分组
HAVING n >= 3             -- 再过滤分组
ORDER BY n DESC           -- 再排序
LIMIT 5;                  -- 最后截断
```

### 5.3 多表连接 — JOIN

这是 SQL 最核心的能力之一。

```sql
-- 假设有两张表
-- users:    id, name
-- orders:   id, user_id, amount, created_at

-- INNER JOIN：只返回两表都匹配的行（交集）
SELECT users.name, orders.amount, orders.created_at
FROM users
INNER JOIN orders ON users.id = orders.user_id;

-- LEFT JOIN：返回左表全部行，右表没匹配的填 NULL
SELECT users.name, orders.amount
FROM users
LEFT JOIN orders ON users.id = orders.user_id;
-- 结果：张三 有订单显示金额，李四 没订单显示 NULL

-- RIGHT JOIN（SQLite 不支持，用 LEFT JOIN + 换表顺序替代）

-- 多表连接
SELECT u.name, o.amount, p.name AS product_name
FROM users u                          -- u 是 users 的别名
INNER JOIN orders o ON u.id = o.user_id       -- o 是 orders 的别名
INNER JOIN products p ON o.product_id = p.id;

-- 自连接（表自己连接自己）
-- 查找同城市的用户对
SELECT a.name, b.name, a.city
FROM users a
INNER JOIN users b ON a.city = b.city AND a.id < b.id;

-- CROSS JOIN（笛卡尔积，每行 × 每行，结果 = 左表行数 × 右表行数）
SELECT * FROM users CROSS JOIN products;  -- 小心：大表会爆炸
```

### 5.4 子查询

```sql
-- 子查询在 WHERE 中
SELECT * FROM users
WHERE age > (SELECT AVG(age) FROM users);

-- 子查询在 FROM 中（必须给别名）
SELECT avg_age FROM (
    SELECT city, AVG(age) AS avg_age FROM users GROUP BY city
) AS city_stats
WHERE avg_age > 30;

-- 子查询在 SELECT 中
SELECT name, age,
    (SELECT AVG(age) FROM users) AS 全体平均年龄,
    age - (SELECT AVG(age) FROM users) AS 与平均的差值
FROM users;

-- 相关子查询（子查询引用外部查询的列）
SELECT name, salary FROM users u
WHERE salary > (
    SELECT AVG(salary) FROM users WHERE city = u.city
);
```

---

## 6. 内置函数大全

### 6.1 聚合函数

| 函数 | 说明 | 示例 |
|------|------|------|
| `COUNT(*)` | 行数 | `SELECT COUNT(*) FROM users` |
| `COUNT(column)` | 非 NULL 值的数量 | `SELECT COUNT(email) FROM users` |
| `COUNT(DISTINCT col)` | 去重计数 | `SELECT COUNT(DISTINCT city) FROM users` |
| `AVG(col)` | 平均值 | `SELECT AVG(age) FROM users` |
| `SUM(col)` | 总和 | `SELECT SUM(amount) FROM orders` |
| `MAX(col)` / `MIN(col)` | 最大/最小值 | `SELECT MAX(salary) FROM users` |
| `GROUP_CONCAT(col, ',')` | 将分组值拼成字符串 | `SELECT city, GROUP_CONCAT(name) FROM users GROUP BY city` |
| `TOTAL(col)` | 总和（始终返回浮点，NULL→0） | `SELECT TOTAL(amount) FROM orders` |

### 6.2 字符串函数

```sql
-- 长度
SELECT LENGTH('hello');          -- 5（字符数）
SELECT LENGTH('你好');           -- 6（UTF-8 字节数）

-- 大小写转换
SELECT UPPER('hello');           -- HELLO
SELECT LOWER('HELLO');           -- hello

-- 截取子串
SELECT SUBSTR('Hello World', 1, 5);   -- Hello（从第1个开始取5个，索引从1开始）
SELECT SUBSTR('Hello World', 7);      -- World（从第7个取到末尾）

-- 替换
SELECT REPLACE('Hello World', 'World', 'SQLite');  -- Hello SQLite

-- 拼接
SELECT 'Hello' || ' ' || 'World';    -- Hello World（用 || 拼接）

-- 去除空白
SELECT TRIM('  hello  ');         -- 'hello'（去两端空白）
SELECT LTRIM('  hello  ');        -- 'hello  '（去左边空白）
SELECT RTRIM('  hello  ');        -- '  hello'（去右边空白）

-- 格式化（PRINTF / FORMAT）
SELECT PRINTF('%.2f', 3.14159);   -- '3.14'
SELECT PRINTF('%s 今年 %d 岁', '张三', 28);  -- '张三 今年 28 岁'

-- 查找位置
SELECT INSTR('Hello World', 'World');  -- 7（找到的位置，找不到返回 0）

-- Unicode 字符
SELECT CHAR(65);                   -- 'A'（ASCII/Unicode 码 → 字符）
SELECT UNICODE('A');              -- 65（字符 → Unicode 码）
```

### 6.3 数字函数

```sql
SELECT ABS(-5);            -- 5（绝对值）
SELECT ROUND(3.14159, 2);  -- 3.14（四舍五入，保留2位）
SELECT ROUND(3.14159);     -- 3.0（四舍五入到整数）
SELECT CEIL(3.14);         -- 4.0（向上取整）
SELECT FLOOR(3.14);        -- 3.0（向下取整）
SELECT RANDOM();           -- 随机整数（每次调用不同）
SELECT RANDOMBLOB(16);     -- 16字节随机 BLOB
SELECT MAX(1, 2, 3);       -- 3（多个值中的最大值）
SELECT MIN(1, 2, 3);       -- 1（多个值中的最小值）
SELECT POWER(2, 10);       -- 1024.0（2的10次方，需要 math 扩展）
SELECT SQRT(16);           -- 4.0（平方根，需要 math 扩展）
SELECT MOD(10, 3);         -- 1（取模，用 % 也可以：10 % 3）
```

### 6.4 日期时间函数

```sql
-- 获取当前时间（UTC）
SELECT DATETIME('now');           -- 2025-06-26 10:30:00
SELECT DATE('now');               -- 2025-06-26
SELECT TIME('now');               -- 10:30:00
SELECT STRFTIME('%Y-%m-%d %H:%M:%S', 'now');  -- 格式化

-- 时间运算（±N days/hours/minutes/...）
SELECT DATETIME('now', '+1 day');       -- 明天
SELECT DATETIME('now', '-7 days');      -- 一周前
SELECT DATETIME('now', '+3 hours');     -- 3小时后
SELECT DATETIME('now', '+1 month');     -- 一个月后
SELECT DATETIME('now', '+1 year');      -- 一年后
SELECT DATETIME('now', 'start of month');  -- 本月第一天
SELECT DATETIME('now', 'start of year');   -- 今年第一天
SELECT DATETIME('now', 'weekday 0');       -- 本周日

-- 计算时间差
SELECT JULIANDAY('2025-12-31') - JULIANDAY('2025-01-01');  -- 364.0 天

-- STRFTIME 格式化符号
-- %Y = 年(4位)   %m = 月(01-12)    %d = 日(01-31)
-- %H = 时(00-23)  %M = 分(00-59)   %S = 秒(00-59)
-- %w = 星期(0-6)  %W = 年第几周     %j = 年第几天
SELECT STRFTIME('%Y年%m月%d日 %H:%M', 'now');   -- 2025年06月26日 14:30
```

### 6.5 条件函数

```sql
-- CASE WHEN ... THEN ... ELSE ... END（SQL 中的 if/else）
SELECT name, age,
    CASE
        WHEN age < 18 THEN '未成年'
        WHEN age < 60 THEN '成年'
        ELSE '老年'
    END AS 年龄段
FROM users;

-- IIF（三元表达式，SQLite 3.32+）
SELECT IIF(age >= 18, '成年', '未成年') FROM users;

-- COALESCE（返回第一个非 NULL 值）
SELECT COALESCE(email, phone, '无联系方式') FROM users;
-- email 为 NULL → 用 phone，phone 也为 NULL → 用 '无联系方式'

-- NULLIF（相等返回 NULL，不相等返回第一个值）
SELECT NULLIF(0, 0);     -- NULL
SELECT NULLIF(5, 0);     -- 5

-- IFNULL（专门处理 NULL 的快捷方式）
SELECT IFNULL(email, '未填写') FROM users;
```

### 6.6 类型转换函数

```sql
SELECT CAST('123' AS INTEGER);   -- 123
SELECT CAST(123 AS TEXT);        -- '123'
SELECT CAST('3.14' AS REAL);     -- 3.14
SELECT TYPEOF(123);              -- 'integer'
SELECT TYPEOF('hello');          -- 'text'
SELECT TYPEOF(NULL);             -- 'null'
```

---

## 7. 表管理 — 约束 / 修改 / 删除

### 7.1 CREATE TABLE — 完整建表语法

```sql
CREATE TABLE IF NOT EXISTS employees (
    -- 主键：自增整数 ID
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- NOT NULL：不允许为空
    name TEXT NOT NULL,

    -- DEFAULT：默认值
    department TEXT DEFAULT '未分配',

    -- UNIQUE：唯一约束（不能有重复值）
    email TEXT UNIQUE,

    -- CHECK：自定义约束（条件不满足则插入失败）
    age INTEGER CHECK(age >= 18 AND age <= 65),

    -- FOREIGN KEY：外键关联
    manager_id INTEGER REFERENCES employees(id),

    -- 组合约束（多列一起唯一）
    UNIQUE(name, department),

    -- 时间戳（自动填充）
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- 如果需要外键生效，必须手动开启（SQLite 默认关闭）
PRAGMA foreign_keys = ON;
```

### 7.2 ALTER TABLE — 修改表结构

SQLite 的 ALTER TABLE 能力有限。

```sql
-- ✅ 支持：重命名表
ALTER TABLE employees RENAME TO staff;

-- ✅ 支持：添加列（只能加到末尾）
ALTER TABLE employees ADD COLUMN phone TEXT;

-- ✅ 支持：重命名列（SQLite 3.25+）
ALTER TABLE employees RENAME COLUMN phone TO mobile;

-- ❌ 不支持：删除列、修改列类型、添加约束
-- 解决办法：创建新表 → 导数据 → 删旧表 → 重命名
-- Step 1: 创建新表结构
CREATE TABLE employees_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    -- 修改后的结构
);
-- Step 2: 导数据
INSERT INTO employees_new (id, name) SELECT id, name FROM employees;
-- Step 3: 删除旧表
DROP TABLE employees;
-- Step 4: 重命名
ALTER TABLE employees_new RENAME TO employees;
```

### 7.3 DROP TABLE — 删除表

```sql
DROP TABLE IF EXISTS employees;  -- 删除表（连数据一起，不可恢复！）
```

### 7.4 约束详解

```sql
-- PRIMARY KEY：主键（唯一 + 非空 + 自动索引）
CREATE TABLE t1 (id INTEGER PRIMARY KEY);  -- 整数主键 = rowid 别名，性能最好

-- UNIQUE：唯一约束
CREATE TABLE t2 (email TEXT UNIQUE);

-- NOT NULL：非空约束
CREATE TABLE t3 (name TEXT NOT NULL);

-- DEFAULT：默认值
CREATE TABLE t4 (
    status TEXT DEFAULT 'active',
    count INTEGER DEFAULT 0
);

-- CHECK：检查约束
CREATE TABLE t5 (
    age INTEGER CHECK(age >= 0 AND age <= 150),
    email TEXT CHECK(email LIKE '%@%')  -- 必须有 @
);

-- FOREIGN KEY：外键约束（需开启 PRAGMA foreign_keys = ON）
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
-- ON DELETE CASCADE: 删用户时自动删订单
-- ON DELETE SET NULL: 删用户时订单的 user_id 设为 NULL
-- ON DELETE RESTRICT: 有订单时不许删用户（默认）
```

---

## 8. 索引与性能优化

### 8.1 什么是索引

索引就像书的**目录**。没有索引 = 从第一页翻到最后找内容（全表扫描）。有索引 = 看目录直接翻到那一页（O(log n)）。

```sql
-- 创建索引
CREATE INDEX idx_users_name ON users(name);
CREATE INDEX idx_users_city_age ON users(city, age);  -- 组合索引
CREATE UNIQUE INDEX idx_users_email ON users(email);   -- 唯一索引

-- 删除索引
DROP INDEX IF EXISTS idx_users_name;

-- 查看查询是否用了索引
EXPLAIN QUERY PLAN SELECT * FROM users WHERE name = '张三';
-- 输出: SEARCH users USING INDEX idx_users_name (name=?)
-- 看到 USING INDEX 说明索引起作用了
```

### 8.2 什么时候建索引

| 场景 | 建议 |
|------|------|
| WHERE 频繁用于筛选的列 | ✅ 建索引 |
| ORDER BY 排序的列 | ✅ 建索引 |
| JOIN 连接的列 | ✅ 建索引 |
| UNIQUE 约束的列 | 自动有索引 |
| PRIMARY KEY | 自动有索引 |
| 经常变化的列 | ⚠️ 慎重（维护索引有成本） |
| 值很少的列（如性别只有男/女） | ❌ 不建（区分度太低，索引无效） |
| 小表（几百行） | ❌ 不需要 |

### 8.3 性能分析命令

```sql
-- 查看查询计划
EXPLAIN QUERY PLAN SELECT * FROM users WHERE age > 30;

-- 查看所有索引
SELECT * FROM sqlite_master WHERE type = 'index';

-- 查看表信息
SELECT * FROM sqlite_master WHERE type = 'table';

-- 开启计时
.timer ON  -- sqlite3 CLI

-- 分析表（更新查询优化器的统计信息）
ANALYZE users;

-- 重建索引（碎片整理）
REINDEX users;
```

### 8.4 常见性能陷阱

```sql
-- ❌ 索引失效：对列做函数运算
SELECT * FROM users WHERE LOWER(name) = 'zhangsan';
-- 解决：用表达式索引或存时转小写

-- ❌ 索引失效：前导模糊查询
SELECT * FROM users WHERE name LIKE '%三';  -- 全表扫描
-- ✅ 只有后置模糊能用索引
SELECT * FROM users WHERE name LIKE '张%';  -- 能用索引

-- ❌ 索引失效：OR 连接不同列
SELECT * FROM users WHERE name = '张' OR email = 'a@b.com';
-- 解决：用 UNION
SELECT * FROM users WHERE name = '张'
UNION
SELECT * FROM users WHERE email = 'a@b.com';

-- ❌ 索引失效：!= 和 NOT IN
SELECT * FROM users WHERE status != 'deleted';  -- 通常不走索引
```

---

## 9. 事务与并发 — ACID / WAL / 锁

### 9.1 什么是事务

事务 = 一组 SQL 操作，要么全部成功，要么全部回滚。保证数据一致性。

```sql
-- 转账示例：A 给 B 转 100 元
BEGIN TRANSACTION;  -- 开始事务
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;
    UPDATE accounts SET balance = balance + 100 WHERE id = 2;
    -- 如果这里出错了...
COMMIT;  -- 提交事务（确认所有修改）
-- 如果出错执行 ROLLBACK; 回滚（撤销所有修改）

-- Python 中的事务
conn.execute("BEGIN")
try:
    conn.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
    conn.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")
    conn.commit()  # 提交
except Exception:
    conn.rollback()  # 回滚
```

### 9.2 ACID 特性

| 特性 | 含义 | SQLite 支持 |
|------|------|-----------|
| Atomicity（原子性） | 操作要么全做要么全不做 | ✅ |
| Consistency（一致性） | 事务前后数据满足所有约束 | ✅ |
| Isolation（隔离性） | 并发事务互不干扰 | ✅ 串行化 |
| Durability（持久性） | 提交后数据不丢失 | ✅ |

### 9.3 WAL 模式（生产必备）

SQLite 默认使用**回滚日志**模式。开启 WAL（Write-Ahead Logging）后，**读和写可以同时进行**。

```sql
-- 查看当前模式
PRAGMA journal_mode;
-- 输出: delete（默认回滚模式）

-- 开启 WAL 模式（持久化设置，重启后仍然有效）
PRAGMA journal_mode = WAL;
-- 输出: wal

-- WAL 模式的优势：
-- 1. 读写可以并发（默认模式写时会锁住读）
-- 2. 写入更快（顺序写 WAL 文件，不直接写主文件）
-- 3. 更安全（崩溃恢复更可靠）
```

```python
# Python 中开启 WAL（连接后第一件事）
conn = sqlite3.connect("demo.db")
conn.execute("PRAGMA journal_mode = WAL")
conn.execute("PRAGMA foreign_keys = ON")  # 外键也要手动开
```

### 9.4 其他重要 PRAGMA

```sql
-- 性能优化
PRAGMA synchronous = NORMAL;    -- 平衡安全与性能（默认 FULL）
PRAGMA cache_size = -8000;      -- 缓存大小（KB，负数为绝对值）
PRAGMA temp_store = MEMORY;     -- 临时表放内存
PRAGMA mmap_size = 268435456;   -- 内存映射 I/O（256MB）

-- 安全相关
PRAGMA foreign_keys = ON;       -- 开启外键约束（必须手动开！）
PRAGMA busy_timeout = 5000;     -- 锁等待超时（毫秒）

-- 信息查询
PRAGMA table_info(users);       -- 查看表结构
PRAGMA index_list(users);       -- 查看索引列表
PRAGMA database_list;           -- 查看所有数据库
```

### 9.5 并发写入处理

```python
# SQLite 同一时间只允许一个连接写入。多线程写需要处理 BusyError
import time
import sqlite3

def execute_with_retry(conn, sql, params=None, max_retries=5):
    """写入失败时自动重试"""
    for attempt in range(max_retries):
        try:
            if params:
                return conn.execute(sql, params)
            return conn.execute(sql)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # 递增等待
                continue
            raise
```

---

## 10. Python sqlite3 模块详解

### 10.1 基础用法速查

```python
import sqlite3

# ---- 连接 ----
conn = sqlite3.connect("demo.db")
conn.row_factory = sqlite3.Row  # 让结果可以按列名访问
conn.execute("PRAGMA journal_mode = WAL")
conn.execute("PRAGMA foreign_keys = ON")

# ---- 执行 SQL ----
cur = conn.cursor()

# 执行一条
cur.execute("SELECT * FROM users WHERE id = ?", (1,))

# 批量执行多条
cur.executemany(
    "INSERT INTO users (name, age) VALUES (?, ?)",
    [("张三", 28), ("李四", 30), ("王五", 25)]
)

# 执行 SQL 脚本（多条语句）
conn.executescript("""
    CREATE TABLE t1(id INTEGER);
    CREATE TABLE t2(id INTEGER);
""")

# ---- 获取结果 ----
cur.execute("SELECT * FROM users")

row = cur.fetchone()       # 取一条，无结果返回 None
rows = cur.fetchall()      # 取所有（大表小心内存！）
rows = cur.fetchmany(100)  # 取 N 条

# 迭代（推荐，不占内存）
for row in cur.execute("SELECT * FROM users"):
    print(dict(row))

# ---- 提交与关闭 ----
conn.commit()   # 提交事务
conn.rollback() # 回滚事务
conn.close()    # 关闭连接
```

### 10.2 row_factory — 自定义结果格式

```python
# 默认：结果返回 tuple → (1, '张三', 28)
conn = sqlite3.connect("demo.db")

# sqlite3.Row：可以用列名访问 → row['name']
conn.row_factory = sqlite3.Row

# dict：直接返回字典 → {'id': 1, 'name': '张三'}
def dict_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
conn.row_factory = dict_factory
```

### 10.3 上下文管理器（自动提交/回滚）

```python
# 用 with 自动提交或回滚
with sqlite3.connect("demo.db") as conn:
    conn.execute("INSERT INTO users (name) VALUES (?)", ("张三",))
    # with 块正常结束 → 自动 commit
    # with 块抛异常 → 自动 rollback

# 注意：with conn 不会自动开启事务！
# 每 execute 自带隐式事务（除非显式 BEGIN）
```

### 10.4 自定义函数和聚合

```python
# 注册自定义函数到 SQL
conn.create_function("add_one", 1, lambda x: x + 1)
conn.execute("SELECT add_one(41)").fetchone()  # (42,)

# 注册正则匹配函数
import re
def regexp(pattern, value):
    return bool(re.search(pattern, str(value)))
conn.create_function("REGEXP", 2, regexp)
conn.execute("SELECT * FROM users WHERE email REGEXP ?", (r'@gmail\.com$',))

# 自定义聚合函数
class Median:
    def __init__(self):
        self.values = []
    def step(self, value):
        self.values.append(value)
    def finalize(self):
        return sorted(self.values)[len(self.values) // 2]

conn.create_aggregate("MEDIAN", 1, Median)
conn.execute("SELECT MEDIAN(age) FROM users")
```

### 10.5 备份

```python
# 在线备份（不阻塞读写的备份方式）
import sqlite3

def backup_db(source_path: str, backup_path: str):
    """热备份数据库"""
    src = sqlite3.connect(source_path)
    dst = sqlite3.connect(backup_path)
    src.backup(dst)
    dst.close()
    src.close()

# 使用
backup_db("demo.db", "demo_backup.db")
```

---

## 11. 进阶特性 — CTE / 窗口函数 / 触发器 / 视图

### 11.1 CTE（公用表表达式）— WITH 子句

CTE 让复杂查询分步写，每步起个名字，比嵌套子查询好读。

```sql
-- 基础 CTE：给子查询命名
WITH high_salary AS (
    SELECT name, department, salary FROM employees WHERE salary > 10000
)
SELECT * FROM high_salary WHERE department = '技术部';

-- 多个 CTE
WITH
    dept_avg AS (
        SELECT department, AVG(salary) AS avg_salary
        FROM employees GROUP BY department
    ),
    top_earners AS (
        SELECT name, department, salary
        FROM employees WHERE salary > 15000
    )
SELECT t.name, t.department, t.salary, d.avg_salary
FROM top_earners t
JOIN dept_avg d ON t.department = d.department;

-- 递归 CTE：生成序列
WITH RECURSIVE cnt(x) AS (
    SELECT 1                -- 初始值
    UNION ALL
    SELECT x + 1 FROM cnt   -- 递推
    WHERE x < 10            -- 终止条件
)
SELECT * FROM cnt;
-- 结果: 1, 2, 3, ..., 10

-- 递归 CTE：树形结构（查所有子节点）
WITH RECURSIVE subordinates(id, name, manager_id, level) AS (
    SELECT id, name, manager_id, 0 FROM employees WHERE id = 1  -- CEO
    UNION ALL
    SELECT e.id, e.name, e.manager_id, s.level + 1
    FROM employees e
    JOIN subordinates s ON e.manager_id = s.id
)
SELECT * FROM subordinates ORDER BY level;
```

### 11.2 窗口函数（SQLite 3.25+）

窗口函数不折叠行（不像 GROUP BY），而是在每行上计算。

```sql
-- ROW_NUMBER：行号
SELECT name, salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS rank
FROM employees;

-- RANK / DENSE_RANK：排名
SELECT name, salary,
    RANK() OVER (ORDER BY salary DESC) AS rank,        -- 1,2,2,4（有间隙）
    DENSE_RANK() OVER (ORDER BY salary DESC) AS dr     -- 1,2,2,3（无间隙）
FROM employees;

-- PARTITION BY：分组内排名
SELECT name, department, salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank
FROM employees;
-- 结果：每个部门内独立排名

-- 累计和 / 移动平均
SELECT date, amount,
    SUM(amount) OVER (ORDER BY date) AS 累计,
    AVG(amount) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS 7天均线
FROM sales;

-- LAG / LEAD：取前一行 / 后一行
SELECT date, amount,
    LAG(amount) OVER (ORDER BY date) AS 前一天,
    amount - LAG(amount) OVER (ORDER BY date) AS 环比变化
FROM sales;
```

### 11.3 触发器（TRIGGER）

触发器 = 当某事件（INSERT/UPDATE/DELETE）发生时自动执行的 SQL。

```sql
-- 自动更新 updated_at 字段
CREATE TRIGGER update_timestamp
    AFTER UPDATE ON users
BEGIN
    UPDATE users SET updated_at = datetime('now') WHERE id = NEW.id;
END;

-- 审计日志：记录所有删除操作
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT,
    action TEXT,
    old_data TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TRIGGER audit_users_delete
    AFTER DELETE ON users
BEGIN
    INSERT INTO audit_log (table_name, action, old_data)
    VALUES ('users', 'DELETE', OLD.name || ',' || OLD.email);
END;

-- 查看触发器
SELECT * FROM sqlite_master WHERE type = 'trigger';

-- 删除触发器
DROP TRIGGER IF EXISTS update_timestamp;
```

### 11.4 视图（VIEW）

视图 = 保存的查询，像一个虚拟表。每次查询视图时重新执行底层 SQL。

```sql
-- 创建视图
CREATE VIEW active_users AS
SELECT id, name, email FROM users WHERE is_active = 1;

-- 像查表一样查视图
SELECT * FROM active_users WHERE name LIKE '张%';

-- 创建视图（带列名）
CREATE VIEW dept_stats(dept, count, avg_salary) AS
SELECT department, COUNT(*), AVG(salary)
FROM employees GROUP BY department;

-- 删除视图
DROP VIEW IF EXISTS active_users;
```

---

## 12. 生产环境最佳实践

### 12.1 连接管理

```python
# ✅ 好的模式：单例连接 + WAL 模式
import sqlite3
from threading import Lock

class Database:
    _instance = None
    _lock = Lock()

    def __new__(cls, db_path="app.db"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._conn = sqlite3.connect(db_path)
                    cls._instance._conn.row_factory = sqlite3.Row
                    cls._instance._conn.execute("PRAGMA journal_mode = WAL")
                    cls._instance._conn.execute("PRAGMA foreign_keys = ON")
                    cls._instance._conn.execute("PRAGMA busy_timeout = 5000")
        return cls._instance

    def execute(self, sql, params=None):
        if params:
            return self._conn.execute(sql, params)
        return self._conn.execute(sql)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()
```

### 12.2 初始化建表

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    token TEXT UNIQUE NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""

def init_db(db_path="app.db"):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
```

### 12.3 数据迁移

```python
# 简单的迁移框架
MIGRATIONS = [
    # (version, sql)
    (1, "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)"),
    (2, "ALTER TABLE users ADD COLUMN email TEXT"),
    (3, "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"),
]

def migrate(db_path="app.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS _migrations (version INTEGER PRIMARY KEY)")

    current = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM _migrations"
    ).fetchone()[0]

    for version, sql in MIGRATIONS:
        if version > current:
            conn.execute(sql)
            conn.execute("INSERT INTO _migrations (version) VALUES (?)", (version,))
            print(f"  已执行迁移 v{version}")

    conn.commit()
    conn.close()
```

### 12.4 安全检查清单

| 检查项 | 做法 |
|--------|------|
| SQL 注入 | ✅ 100% 使用参数化查询 `?`，不拼接字符串 |
| 外键约束 | ✅ `PRAGMA foreign_keys = ON` |
| WAL 模式 | ✅ `PRAGMA journal_mode = WAL` |
| 锁超时 | ✅ `PRAGMA busy_timeout = 5000` |
| 定期备份 | ✅ 每天复制 `.db` 文件 |
| 敏感数据 | ✅ 加密存储（应用层 AES） |
| 并发安全 | ✅ 一个连接一个线程，多线程用队列 |

---

## 13. 常见错误与反模式

### 13.1 忘记开启外键约束

```python
# ❌ 错误：删除用户不会删除订单
conn = sqlite3.connect("app.db")
conn.execute("DELETE FROM users WHERE id = 1")

# ✅ 正确：先开启外键
conn.execute("PRAGMA foreign_keys = ON")
conn.execute("DELETE FROM users WHERE id = 1")
```

### 13.2 字符串拼接导致 SQL 注入

```python
# ❌ 危险
name = "张三'; DROP TABLE users; --"
conn.execute(f"SELECT * FROM users WHERE name = '{name}'")

# ✅ 安全
conn.execute("SELECT * FROM users WHERE name = ?", (name,))
```

### 13.3 忘记 COMMIT 导致数据丢失

```python
# ❌ 数据没保存
conn.execute("INSERT INTO users (name) VALUES ('张三')")
# 没有 conn.commit()，连接关闭后数据丢失

# ✅ 正确
with sqlite3.connect("app.db") as conn:
    conn.execute("INSERT INTO users (name) VALUES (?)", ("张三",))
    # with 块正常结束自动 commit
```

### 13.4 默认模式下读写互斥

```python
# 问题：一个连接在写，其他连接读也会被阻塞
# 解决：开启 WAL 模式
conn.execute("PRAGMA journal_mode = WAL")
```

### 13.5 大结果集直接 fetchall() 爆内存

```python
# ❌ 100万行全读进内存
rows = conn.execute("SELECT * FROM big_table").fetchall()

# ✅ 逐行迭代
for row in conn.execute("SELECT * FROM big_table"):
    process(row)
```

### 13.6 在循环中逐条 INSERT

```python
# ❌ 慢（每条都要走一遍 SQL 解析和执行）
for item in data:
    conn.execute("INSERT INTO t VALUES (?)", (item,))

# ✅ 快（批量执行）
conn.executemany("INSERT INTO t VALUES (?)", [(item,) for item in data])

# ✅ 更快（在事务中批量执行）
with conn:
    conn.executemany("INSERT INTO t VALUES (?)", [(item,) for item in data])
```

---

## 14. 企业级封装示例

以下是可以在项目中直接用的 `Database` 类，包含：WAL 模式、自动迁移、重试机制、上下文管理器。

```python
import sqlite3
import time
from pathlib import Path

class Database:
    """
    企业级 SQLite 数据库封装

    使用示例:
        db = Database("app.db", schema=SCHEMA, migrations=MIGRATIONS)
        db.execute("INSERT INTO users (name) VALUES (?)", ("张三",))
        users = db.fetchall("SELECT * FROM users WHERE age > ?", (25,))

    特性:
        - WAL 模式（读写并发）
        - 自动外键约束
        - 自动迁移
        - 写入失败自动重试
        - 上下文管理器（自动提交/回滚）
    """

    def __init__(self, db_path: str, schema: str = "", migrations: list = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))

        # 生产配置
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA synchronous = NORMAL")

        # 初始化
        if schema:
            self._conn.executescript(schema)
        if migrations:
            self._run_migrations(migrations)

    # ========== 基础操作 ==========

    def execute(self, sql: str, params=None, retries: int = 3):
        """执行 SQL（INSERT/UPDATE/DELETE），失败自动重试"""
        for attempt in range(retries):
            try:
                if params:
                    return self._conn.execute(sql, params)
                return self._conn.execute(sql)
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < retries - 1:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise

    def executemany(self, sql: str, params_list: list):
        return self._conn.executemany(sql, params_list)

    def fetchone(self, sql: str, params=None) -> dict | None:
        """查询单行，返回 dict 或 None"""
        cur = self.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def fetchall(self, sql: str, params=None) -> list[dict]:
        """查询多行，返回 dict 列表"""
        cur = self.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def fetch_column(self, sql: str, params=None, column: int = 0) -> list:
        """查询单列"""
        cur = self.execute(sql, params)
        return [row[column] for row in cur.fetchall()]

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    # ========== 迁移 ==========

    def _run_migrations(self, migrations: list):
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS _migrations (version INTEGER PRIMARY KEY)"
        )
        current = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM _migrations"
        ).fetchone()[0]

        for version, sql in migrations:
            if version > current:
                self._conn.execute(sql)
                self._conn.execute(
                    "INSERT INTO _migrations (version) VALUES (?)", (version,)
                )
        self._conn.commit()

    # ========== 工具方法 ==========

    def table_exists(self, table: str) -> bool:
        """检查表是否存在"""
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None

    def get_tables(self) -> list[str]:
        """列出所有表"""
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r["name"] for r in rows]

    def get_columns(self, table: str) -> list[dict]:
        """获取表结构"""
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [dict(r) for r in rows]

    def vacuum(self):
        """压缩数据库（回收删除数据后的空闲空间）"""
        self._conn.execute("VACUUM")

    def backup(self, backup_path: str):
        """热备份"""
        dst = sqlite3.connect(backup_path)
        self._conn.backup(dst)
        dst.close()

    def close(self):
        self._conn.close()

    # ========== 上下文管理器 ==========

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        # 不在 exit 中 close，允许复用

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass


# ============================================================
# 使用示例
# ============================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    age INTEGER CHECK(age >= 0),
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);
"""

MIGRATIONS = [
    (1, "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)"),
    (2, "ALTER TABLE users ADD COLUMN email TEXT"),
    (3, "ALTER TABLE users ADD COLUMN age INTEGER"),
    (4, "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"),
]


if __name__ == "__main__":
    # 初始化
    db = Database("enterprise.db", schema=SCHEMA, migrations=MIGRATIONS)

    # 插入
    db.execute(
        "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
        ("张三", "zhang@example.com", 28),
    )
    db.commit()

    # 查询
    user = db.fetchone("SELECT * FROM users WHERE email = ?", ("zhang@example.com",))
    print(user)  # {'id': 1, 'name': '张三', ...}

    # 列表
    all_users = db.fetchall("SELECT * FROM users ORDER BY created_at DESC")
    print(all_users)

    # 使用上下文管理器
    with db:
        db.execute("UPDATE users SET age = ? WHERE id = ?", (29, 1))
    # with 块正常结束 → 自动 commit

    # 工具方法
    print(db.get_tables())
    print(db.get_columns("users"))
    db.backup("enterprise_backup.db")

    db.close()
    print("✅ 运行完成")
```

---

## 快速参考卡

### 最常用的 15 个操作

```python
import sqlite3

# 1. 连接 + 配置
conn = sqlite3.connect("db.sqlite")
conn.row_factory = sqlite3.Row
conn.execute("PRAGMA journal_mode = WAL")
conn.execute("PRAGMA foreign_keys = ON")

# 2. 建表
conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)")

# 3. 插入
conn.execute("INSERT INTO t (name) VALUES (?)", ("张三",))
conn.commit()

# 4. 批量插入
conn.executemany("INSERT INTO t (name) VALUES (?)", [("A",), ("B",)])

# 5. 查询
rows = conn.execute("SELECT * FROM t WHERE name = ?", ("张三",)).fetchall()

# 6. 更新
conn.execute("UPDATE t SET name = ? WHERE id = ?", ("李四", 1))

# 7. 删除
conn.execute("DELETE FROM t WHERE id = ?", (1,))

# 8. 计数
count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]

# 9. LIKE 模糊查询
conn.execute("SELECT * FROM t WHERE name LIKE ?", ("%张%",))

# 10. 检查是否存在
exists = conn.execute("SELECT 1 FROM t WHERE id = ?", (1,)).fetchone() is not None

# 11. 分组统计
conn.execute("SELECT city, COUNT(*) FROM users GROUP BY city").fetchall()

# 12. 排序分页
conn.execute("SELECT * FROM t ORDER BY id DESC LIMIT 10 OFFSET 20").fetchall()

# 13. 连接查询
conn.execute("SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id").fetchall()

# 14. 事务
conn.execute("BEGIN")
# ... 多条 SQL ...
conn.commit()  # 或 conn.rollback()

# 15. 关闭
conn.close()
```

---

> 全文参考：SQLite 官方文档 (https://sqlite.org/docs.html) | Python sqlite3 文档 (https://docs.python.org/3/library/sqlite3.html)