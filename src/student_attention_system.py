import functools
import os
from flask import *
from werkzeug.utils import secure_filename
from src.dbconnection import *

app = Flask(__name__)
app.secret_key="1234"

def login_required(func):
    @functools.wraps(func)
    def secure_function():
        if "lid" not in session:
            return render_template('login_index.html')
        return func()
    return secure_function

@app.route('/')
def log():
    return render_template('login_index.html')

@app.route('/logincode',methods=['post'])
def logincode():
    un=request.form['textfield']
    ps=request.form['textfield2']
    a="SELECT * FROM `login` WHERE `username`=%s AND `password`=%s"
    val=(un,ps)
    s=selectone(a,val)
    if s is None:
        return '''<script>alert("invalid username or password");window.location='/'</script>'''
    elif s['type']=="admin":
        session['lid']=s['login_id']
        return '''<script>alert("Welcome Admin");window.location='/admin_home'</script>'''
    elif s['type'] == "staff":
        session['lid'] = s['login_id']
        return '''<script>alert("Welcome Staff");window.location='/staff_home'</script>'''
    elif s['type'] == "student":
        session['lid'] = s['login_id']
        return '''<script>alert("Welcome Student");window.location='/student_home'</script>'''
    else:
        return '''<script>alert("invalid!!!!");window.location='/'</script>'''

@app.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect('/')

#==================================================ADMIN================================================================
@app.route('/admin_home')
@login_required
def admin_home():
    return render_template('admin/admin_index.html')

@app.route('/view_course')
@login_required
def view_course():
    q="select * from course"
    res=selectall(q)
    print(res)
    return render_template('admin/view_course.html',data=res)

@app.route('/add_course')
@login_required
def add_course():
    return render_template('admin/add_course.html')

@app.route('/add_course_post', methods=['POST'])
@login_required
def add_course_post():
    course=request.form['textfield']
    q="INSERT INTO `course` VALUES (NULL,%s)"
    iud(q,course)
    return '''<script>alert("Add successfully");window.location='/view_course#about'</script>'''

@app.route('/delete_course')
@login_required
def delete_course():
    cid=request.args.get('id')
    q="DELETE FROM `course` WHERE `cid`=%s"
    iud(q,cid)
    return '''<script>alert("Deleted successfully");window.location='/view_course#about'</script>'''

@app.route('/view_staff')
@login_required
def view_staff():
    q="select * from staff"
    res=selectall(q)
    return render_template('admin/view_staff.html',data=res)

@app.route('/add_staff')
@login_required
def add_staff():
    return render_template('admin/add_staff.html')

@app.route('/add_staff_post', methods=['POST'])
@login_required
def add_staff_post():
    name=request.form['textfield']
    place=request.form['textfield2']
    phone=request.form['textfield3']
    email=request.form['textfield4']
    uname=request.form['textfield5']
    psw=request.form['textfield6']
    q="INSERT INTO `login` VALUES (NULL,%s,%s,'staff')"
    val=(uname,psw)
    res=iud(q,val)

    qry="INSERT INTO `staff` VALUES (NULL,%s,%s,%s,%s,%s)"
    val1=(str(res),name,place,phone,email)
    iud(qry,val1)
    return '''<script>alert("Added successfully");window.location='/view_staff#about'</script>'''

@app.route('/edit_staff')
@login_required
def edit_staff():
    id=request.args.get('id')
    session['sid']=id
    q="select * from staff where staff_lid=%s"
    res=selectone(q,id)
    print(res)
    return render_template('admin/edit_staff.html',data=res)

@app.route('/edit_staff_post', methods=['POST'])
@login_required
def edit_staff_post():
    name=request.form['textfield']
    place=request.form['textfield2']
    phone=request.form['textfield3']
    email=request.form['textfield4']
    q="UPDATE `staff` SET `name`=%s,`place`=%s,`phone`=%s,`email`=%s WHERE `staff_lid`=%s"
    val=(name,place,phone,email,str(session['sid']))
    res=iud(q,val)
    return '''<script>alert("Updated  successfully");window.location='/view_staff#about'</script>'''

@app.route('/delete_staff')
@login_required
def delete_staff():
    cid=request.args.get('id')
    q="DELETE FROM `staff` WHERE `staff_lid`=%s"
    iud(q,cid)
    qry="delete from login where login_id=%s"
    iud(qry,cid)
    return '''<script>alert("Deleted successfully");window.location='/view_staff#about'</script>'''

@app.route('/view_student')
@login_required
def view_student():
    q="SELECT `student`.*,`course`.* FROM `student` JOIN `course` ON `student`.`course_id`=`course`.`cid`"
    res=selectall(q)
    return render_template('admin/view_student.html',data=res)

@app.route('/add_student')
@login_required
def add_student():
    q="select * from course"
    res=selectall(q)
    return render_template('admin/add_student.html',data=res)

@app.route('/add_stud_post', methods=['POST'])
@login_required
def add_stud_post():
    name=request.form['textfield']
    gender=request.form['radiobutton']
    place=request.form['textfield2']
    post=request.form['textfield22']
    phone=request.form['textfield3']
    email=request.form['textfield4']
    course=request.form['select2']
    uname=request.form['textfield5']
    psw=request.form['textfield6']

    q="INSERT INTO `login` VALUES (NULL,%s,%s,'student')"
    val=(uname,psw)
    res=iud(q,val)

    qry="INSERT INTO `student` VALUES (NULL,%s,%s,%s,%s,%s,%s,%s,%s)"
    val1=(str(res),course,name,gender,place,post,phone,email)
    iud(qry,val1)
    return '''<script>alert("Added successfully");window.location='/view_student#about'</script>'''

@app.route('/edit_student')
@login_required
def edit_student():
    id=request.args.get('id')
    session['stud_id']=id
    qry="SELECT * FROM `student` WHERE `lid`=%s"
    res1=selectone(qry,id)
    q="select * from course"
    res=selectall(q)
    return render_template('admin/edit_student.html',data=res,stud=res1)

@app.route('/edit_stud_post', methods=['POST'])
@login_required
def edit_stud_post():
    name=request.form['textfield']
    gender=request.form['radiobutton']
    place=request.form['textfield2']
    post=request.form['textfield22']
    phone=request.form['textfield3']
    email=request.form['textfield4']
    course=request.form['select2']
    qry="UPDATE `student` SET `course_id`=%s,`name`=%s,`gender`=%s,`place`=%s,`post`=%s,`phone`=%s,`email`=%s WHERE `lid`=%s"
    val1=(course,name,gender,place,post,phone,email,session['stud_id'])
    iud(qry,val1)
    return '''<script>alert("Updated successfully");window.location='/view_student#about'</script>'''

@app.route('/delete_student')
@login_required
def delete_student():
    id=request.args.get('id')
    q="DELETE FROM `student` WHERE `lid`=%s"
    iud(q,id)
    qry="delete from login where login_id=%s"
    iud(qry,id)
    return '''<script>alert("Deleted successfully");window.location='/view_student#about'</script>'''


@app.route('/view_sub')
@login_required
def view_sub():
    q="SELECT `subject`.*,`course`.* FROM `course` JOIN `subject` ON `course`.`cid`=`subject`.`courde_id`"
    res=selectall(q)
    return render_template('admin/view_sub.html',data=res)

@app.route('/add_sub')
@login_required
def add_sub():
    q="select * from course"
    res=selectall(q)
    return render_template('admin/add_sub.html',data=res)

@app.route('/add_sub_post', methods=['POST'])
@login_required
def add_sub_post():
    course=request.form['select']
    sub=request.form['textfield2']
    sem=request.form['select2']
    q="INSERT INTO `subject` VALUES (NULL,%s,%s,%s)"
    val=(course,sub,sem)
    iud(q,val)
    return '''<script>alert("Add successfully");window.location='/view_sub#about'</script>'''

@app.route('/edit_sub')
@login_required
def edit_sub():
    id=request.args.get('id')
    session['subid']=id
    qry="SELECT * FROM `subject` WHERE `sub_id`=%s"
    res1=selectone(qry,id)
    q="select * from course"
    res=selectall(q)
    return render_template('admin/edit_sub.html',data=res,val=res1)

@app.route('/edit_sub_post', methods=['POST'])
@login_required
def edit_sub_post():
    print(request.form)
    course=request.form['select']
    sub=request.form['textfield2']
    sem=request.form['select2']
    q="UPDATE `subject` SET `courde_id`=%s,`subject`=%s,`semester`=%s WHERE `sub_id`=%s"
    val=(course,sub,sem,session['subid'])
    iud(q,val)
    return '''<script>alert("Updated successfully");window.location='/view_sub#about'</script>'''

@app.route('/delete_sub')
@login_required
def delete_sub():
    cid=request.args.get('id')
    q="DELETE FROM `subject` WHERE `sub_id`=%s"
    iud(q,cid)
    return '''<script>alert("Deleted successfully");window.location='/view_sub#about'</script>'''

@app.route('/sub_allocation')
@login_required
def sub_allocation():
    q="select * from subject"
    res=selectall(q)
    qq="select * from staff"
    res1=selectall(qq)
    return render_template('admin/sub_allocation.html',sub=res,staff=res1)

@app.route('/sub_allo_post', methods=['POST'])
@login_required
def sub_allo_post():
    sub=request.form['select']
    staff=request.form['select2']
    q="insert into `sub_allocation` values (null,%s,%s)"
    val=(sub,staff)
    iud(q,val)
    return '''<script>alert("Allocated successfully");window.location='/sub_allocation#about'</script>'''



@app.route('/view_staff_perfomance')
@login_required
def view_staff_perfomance():
    q="SELECT `sub_allocation`.*,`staff`.`name`,`phone`,`staff`.`staff_id`,AVG(`rating`.`score`) AS avgscr FROM `rating` JOIN `sub_allocation` ON `rating`.`sub_id`=`sub_allocation`.`sub_id` JOIN `staff` ON `staff`.`staff_lid`=`sub_allocation`.`staff_id` GROUP BY `staff`.`staff_id`"
    res=selectall(q)
    return render_template('admin/view_staff_perfomance.html',data=res)


#=====================================================STAFF=============================================================


@app.route('/staff_home')
@login_required
def staff_home():
    return render_template('staff/staff_index.html')

@app.route('/view_allocated_sub')
@login_required
def view_allocated_sub():
    q="SELECT `sub_allocation`.*,`subject`.* FROM `sub_allocation` JOIN `subject` ON `sub_allocation`.`sub_id`=`subject`.`sub_id` WHERE `sub_allocation`.`staff_id`=%s"
    res=selectall2(q,str(session['lid']))
    return render_template('staff/view_allocated_sub.html',data=res)

@app.route('/view_doubt')
@login_required
def view_doubt():
    q="select `doubt`.*,`student`.* from `doubt` join `student` on `doubt`.`student_id`=`student`.`lid` where `doubt`.`staff_id`=%s"
    res=selectall2(q,str(session['lid']))
    return render_template('staff/view_doubt.html',data=res)

@app.route('/clear_doubt')
@login_required
def clear_doubt():
    id=request.args.get('id')
    session['did']=id
    return render_template('staff/clear_doubt.html')

@app.route('/reply_doubt', methods=['POST'])
@login_required
def reply_doubt():
    rply=request.form['textfield']
    q="UPDATE `doubt` SET `reply`=%s WHERE `doubt_id`=%s"
    val=(rply,session['did'])
    iud(q,val)
    return '''<script>alert("Cleared successfully");window.location='/view_doubt#about'</script>'''

@app.route('/view_notes')
@login_required
def view_notes():
    q="select `note`.*,`subject`.* from `note` join `subject` on `note`.`sub_id`=`subject`.`sub_id` where `note`.`staff_id`=%s"
    res=selectall2(q,session['lid'])
    return render_template('staff/view_notes.html',data=res)

@app.route('/add_notes')
@login_required
def add_notes():
    q="SELECT `sub_allocation`.*,`subject`.* FROM `sub_allocation` JOIN `subject` ON `sub_allocation`.`sub_id`=`subject`.`sub_id` WHERE `sub_allocation`.`staff_id`=%s"
    res=selectall2(q,str(session['lid']))
    return render_template('staff/add_notes.html',data=res)

@app.route('/add_note_post', methods=['POST'])
@login_required
def add_note_post():
    sub=request.form['select']

    image = request.files['file']
    filename = secure_filename(image.filename)
    image.save(os.path.join('static/notes', filename))

    q="INSERT INTO `note` VALUES (NULL,%s,%s,%s,CURDATE())"
    val=(sub,session['lid'],filename)
    iud(q,val)
    return '''<script>alert("Added successfully");window.location='/view_notes#about'</script>'''

@app.route('/delete_note')
@login_required
def delete_note():
    cid=request.args.get('id')
    q="DELETE FROM `note` WHERE `note_id`=%s"
    iud(q,cid)
    return '''<script>alert("Deleted successfully");window.location='/view_notes#about'</script>'''

@app.route('/view_student_result')
@login_required
def view_student_result():
    q="SELECT `result`.*,`student`.*,`subject`.*,`sub_allocation`.* FROM `result` JOIN `student` ON `result`.`stud_id`=`student`.`lid` JOIN `subject` ON `result`.`sub_id`=`subject`.`sub_id` JOIN `sub_allocation` ON `sub_allocation`.`sub_id`=`result`.`sub_id` WHERE `sub_allocation`.`staff_id`=%s"
    res=selectall2(q,session['lid'])
    return render_template('staff/view_student_result.html',data=res)

@app.route('/add_student_result')
@login_required
def add_student_result():
    q="select `student`.*,`sub_allocation`.*,`subject`.* from `student` join `subject` on `subject`.`courde_id`=`student`.`course_id` join `sub_allocation` on `sub_allocation`.`sub_id`=`subject`.`sub_id` where `sub_allocation`.`staff_id`=%s"
    res=selectall2(q,session['lid'])
    return render_template('staff/add_student_result.html',data=res)

@app.route('/add_mark', methods=['POST'])
@login_required
def add_mark():
    stud=request.form['select2']
    sub=request.form['select']
    mark=request.form['textfield']
    q="INSERT INTO `result` VALUES (NULL,%s,%s,%s)"
    val=(stud,sub,mark)
    iud(q,val)
    return '''<script>alert("Added successfully");window.location='/view_student_result#about'</script>'''

#====================================================STUDENT============================================================


@app.route('/student_home')
@login_required
def student_home():
    return render_template('student/student_index.html')

@app.route('/ask_doubt')
@login_required
def ask_doubt():
    q="SELECT `subject`.*,`student`.*,`sub_allocation`.*,`staff`.* FROM `subject` JOIN `student` ON `student`.`course_id`=`subject`.`courde_id` JOIN `sub_allocation` ON `sub_allocation`.`sub_id`=`subject`.`sub_id` JOIN `staff` ON `staff`.`staff_lid`=`sub_allocation`.`staff_id` WHERE `student`.`lid`=%s"
    res=selectall2(q,session['lid'])
    return render_template('student/ask_doubt.html',data=res)

@app.route('/ask_doubt_post', methods=['POST'])
@login_required
def ask_doubt_post():
    staff=request.form['select']
    doubt=request.form['textfield']
    q="INSERT INTO `doubt` VALUES (NULL,%s,%s,%s,'pending',CURDATE())"
    val=(str(session['lid']),staff,doubt)
    iud(q,val)
    return '''<script>alert("Asked successfully");window.location='/view_doubt_reply#about'</script>'''

@app.route('/view_doubt_reply')
@login_required
def view_doubt_reply():
    q="SELECT `doubt`.*,`staff`.* FROM `doubt` JOIN `staff` ON `doubt`.`staff_id`=`staff`.`staff_lid`"
    res=selectall(q)
    return render_template('student/view_doubt_reply.html',data=res)

@app.route('/stud_view_notes')
@login_required
def stud_view_notes():
    q="SELECT `note`.*,`sub_allocation`.*,`subject`.*,`student`.`course_id` FROM `note` JOIN `sub_allocation` ON `note`.`staff_id`=`sub_allocation`.`staff_id` JOIN `subject` ON `subject`.`sub_id`=`sub_allocation`.`sub_id` JOIN `student` ON `student`.`course_id`=`subject`.`courde_id` WHERE `student`.`lid`=%s"
    res=selectall2(q,session['lid'])
    return render_template('student/view_notes.html',data=res)

@app.route('/stud_view_subjects')
@login_required
def stud_view_subjects():
    q="SELECT `subject`.*,`student`.* FROM `subject` JOIN `student` ON `student`.`course_id`=`subject`.`courde_id` WHERE `student`.`lid`=%s"
    res=selectall2(q,str(session['lid']))
    return render_template('student/view_subjects.html',data=res)

@app.route('/view_subject_rating')
@login_required
def view_subject_rating():
    q="SELECT `subject`.*,AVG(`rating`.`score`) AS avgscr FROM `rating` JOIN `subject` ON `rating`.`sub_id`=`subject`.`sub_id` WHERE `subject`.`sub_id` IN(SELECT `sub_id` FROM `sub_allocation` WHERE `staff_id`=%s) GROUP BY `rating`.`sub_id`"
    res=selectall2(q,session['lid'])
    return render_template('staff/vierw_sub_rating.html',data=res)


app.run(debug=True)
