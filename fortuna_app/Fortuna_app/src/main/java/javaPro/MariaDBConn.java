package javaPro;

import java.sql.Connection;
import java.sql.DriverManager;

public class MariaDBConn {

        public static Connection getConnection() {

                try {

        			String dbURL = "jdbc:mariadb://localhost:3306/rkdrb0070";
        			String dbID = "rkdrb0070";
        			String dbPassword = "Krkdrb0070";
                    Class.forName("org.mariadb.jdbc.Driver");
                    return DriverManager.getConnection(dbURL, dbID, dbPassword);

                }catch (Exception e) {
                    // TODO: handle exception
                    e.printStackTrace();
                }


                return null;
        }
}