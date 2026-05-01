#继承允许你创建「子类」，复用「父类」的属性和方法，还能扩展自己的功能。
#比如：「学生」是「人」的子类，继承人的 “姓名、年龄”，又新增 “学号、成绩” 属性。
# 父类（基类）：Person
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"我叫{self.name}，今年{self.age}岁。")

# 子类（派生类）：Student 继承 Person（语法：class 子类(父类):）
class Student(Person):
    # 子类的初始化方法：先调用父类的__init__，再添加自己的属性(student_id和score)
    def __init__(self, name, age, student_id, score):
        # 调用父类的初始化方法（必须先做）调用父类的__init__，初始化父类的name和age
        super().__init__(name, age)
        # 子类新增的属性
        self.student_id = student_id
        self.score = score

    # 子类扩展自己的方法
    def show_score(self):
        print(f"{self.name}的学号是{self.student_id}，成绩是{self.score}分。")

    # 子类重写父类的方法（覆盖父类逻辑）
    def introduce(self):
        print(f"我叫{self.name}，今年{self.age}岁，学号{self.student_id}。")

# 创建子类对象
stu = Student("王五", 18, "2024001", 95)

# 调用继承自父类的属性
print(stu.name)  # 输出：王五

# 调用子类自己的方法
stu.show_score()  # 输出：王五的学号是2024001，成绩是95分。

# 调用重写后的方法（执行子类的逻辑，而非父类）
stu.introduce()  # 输出：我叫王五，今年18岁，学号2024001。