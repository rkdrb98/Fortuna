<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>

<!DOCTYPE html>

<html lang="ko">

<head>
    <!--          meta 선언          -->
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <!--          link 선언          -->
    <link rel="stylesheet" href="../css/style_test_css.css">
    <link rel="stylesheet" href="../css/join_test_css.css">

    <!--          script 선언          -->
    <script src="https://kit.fontawesome.com/e1bd1cb2a5.js"></script>
    <script src="https://code.jquery.com/jquery-3.4.1.js"></script>

    <script src="../js/script.js"></script>

    <title>
        여기가조 회원가입!
    </title>
</head>

<body>
    

    <div class="join_container">
        <h2>
            회원 가입
        </h2>
        <form method="post" action="./join_action.jsp">
            <h3>아이디</h3>
            <div class="joinID">
                <input type="text" class="input" style="ime-mode:disabled;" placeholder="아이디" name="userID" title="아이디" maxlength="20">
            </div>
            <h3>비밀번호</h3>
            <div class="joinPassword">
                <input type="password" class="input" placeholder="비밀번호" name="userPassword" title="비밀번호" maxlength="20">
            </div>

            </div>
            <input type="submit" class="bt_join" value="회원가입">
        </form>
    </div>


</body>
  <footer>
        <div class="footer_container">
            <div class="footA">
                2팀
            </div>
            <div class="footB">
                Copyright 여기가조 Rights Reserved.
            </div>
        </div>
    </footer> 
</html>