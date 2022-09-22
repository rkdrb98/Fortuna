package javaPro;

public class inquiry {

	private int inquiryID;
	private String Vehicle_type;
	private String region_name;
	private String date;
	private String vehicle_speed;
	private String speed_limit;
	private int Available;
	
	public int getinquiryID() {
		return inquiryID;
	}

	public void setinquiryID(int inquiry) {
		this.inquiryID = inquiry;
	}

	
	
	
	public String getVehicle_type() {
		return Vehicle_type;
	}

	public void setVehicle_type(String Vehicle_type) {
		this.Vehicle_type = Vehicle_type;
	}

	
	
	
	public String getregion_name() {
		return region_name;
	}

	public void setregion_name(String region_name) {
		this.region_name = region_name;
	}

	
	
	
	public String getDate() {
		return date;
	}

	public void setDate(String date) {
		this.date = date;
	}

	
	
	
	public String getvehicle_speed() {
		return vehicle_speed;
	}

	public void setvehicle_speed(String vehicle_speed) {
		this.vehicle_speed= vehicle_speed;
	}
	
	
	
	
	public String getspeed_limit() {
		return speed_limit;
	}

	public void setspeed_limit(String speed_limit) {
		this.speed_limit= speed_limit;
	}
	
	
	
	
	public int getAvailable() {
		return Available;
	}
	public void setAvailable(int Available) {
		this.Available = Available;
	}
}