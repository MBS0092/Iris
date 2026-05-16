/*
  IRIS RECOGNITION SYSTEM — TFT DISPLAY

  FLOW:
    1. ESP32 sends frame → Python detects iris
    2. KNOWN   → TFT home shows GREEN "Access Granted + Name"
    3. UNKNOWN → TFT shows yellow confirm screen:
                 "Unknown iris. Enroll this person?"
                 [YES ENROLL]  [NO IGNORE]
    4. Tap YES  → keyboard appears → type name → ENROLL
    5. Python enrolls from saved frame → TFT confirms

  Serial  Python → Arduino:
    UI|STATUS|known
    UI|NAME|Ahmad
    UI|CONF|0.923
    UI|MESSAGE|Access Granted
    UI|ENROLL_PROMPT|         <- new unknown iris, ask user
    UI|ENROLL_OK|Ahmad
    UI|ENROLL_FAIL|reason

  Serial  Arduino → Python:
    ENROLL:Ahmad
    IGNORE
    GETUI
*/

#include <MCUFRIEND_kbv.h>
#include <Adafruit_GFX.h>
#include <TouchScreen.h>

#define YP A3
#define XM A2
#define YM 9
#define XP 8

#define TS_MINX 120
#define TS_MAXX 900
#define TS_MINY 70
#define TS_MAXY 920
#define MINPRESSURE 120
#define MAXPRESSURE 1000

MCUFRIEND_kbv tft;
TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);

#define SW 480
#define SH 320

#define BLACK   0x0000
#define WHITE   0xFFFF
#define GRAY    0x4208
#define LGRAY   0x8410
#define BLUE    0x001F
#define DBLUE   0x000D
#define RED     0xF800
#define DRED    0x7800
#define GREEN   0x07E0
#define DGREEN  0x03C0
#define YELLOW  0xFFE0
#define DYELLOW 0x8C00
#define CYAN    0x07FF
#define ORANGE  0xFC00
#define NAVY    0x000F
#define TEAL    0x0410

// ── State ──
String lastStatus  = "waiting";
String lastName    = "";
String lastConf    = "0.000";
String lastMessage = "Waiting for scan...";
String rxLine      = "";
String enrollName  = "";

String   notifyMsg   = "";
uint16_t notifyCol   = WHITE;
unsigned long notifyUntil = 0;

enum Screen { HOME, CONFIRM_ENROLL, KEYBOARD };
Screen screen = HOME;

// ── Button helper ──
struct Btn { int x, y, w, h; };
static bool hit(int tx, int ty, const Btn& b) {
  return tx>=b.x && tx<b.x+b.w && ty>=b.y && ty<b.y+b.h;
}
static void fillBtn(const Btn& b, uint16_t bg, uint16_t border,
                    const char* lbl, uint16_t tc, uint8_t sz=2) {
  tft.fillRoundRect(b.x,b.y,b.w,b.h,5,bg);
  tft.drawRoundRect(b.x,b.y,b.w,b.h,5,border);
  tft.setTextSize(sz); tft.setTextColor(tc,bg);
  int16_t x1,y1; uint16_t tw,th;
  tft.getTextBounds(lbl,0,0,&x1,&y1,&tw,&th);
  tft.setCursor(b.x+(b.w-tw)/2, b.y+(b.h-th)/2);
  tft.print(lbl);
}

// ════════════════════════════════════════
// HOME SCREEN
// ════════════════════════════════════════
Btn hBtn_ManualEnroll = { 20,  262, 200, 46 };
Btn hBtn_Clear        = { 260, 262, 200, 46 };

static uint16_t statusColor() {
  if (lastStatus=="known")    return GREEN;
  if (lastStatus=="unknown")  return YELLOW;
  if (lastStatus=="error")    return RED;
  if (lastStatus=="enrolled") return GREEN;
  return LGRAY;
}

static void drawHomeHeader() {
  tft.fillRect(0,0,SW,46,NAVY);
  tft.drawFastHLine(0,46,SW,CYAN);
  tft.setTextSize(3); tft.setTextColor(CYAN,NAVY);
  tft.setCursor(14,10); tft.print("IRIS ACCESS");
  tft.setTextSize(1); tft.setTextColor(LGRAY,NAVY);
  tft.setCursor(SW-100,18); tft.print("BIOMETRIC SYS");
}

static void drawStatusCard() {
  uint16_t col = statusColor();
  tft.fillRoundRect(14,54,SW-28,96,6,DBLUE);
  tft.drawRoundRect(14,54,SW-28,96,6,col);

  String st = lastStatus; st.toUpperCase();
  tft.setTextSize(3); tft.setTextColor(col,DBLUE);
  int16_t x1,y1; uint16_t tw,th;
  tft.getTextBounds(st.c_str(),0,0,&x1,&y1,&tw,&th);
  tft.setCursor(14+(SW-28-tw)/2,64); tft.print(st);

  tft.setTextSize(2); tft.setTextColor(WHITE,DBLUE);
  String msg = lastMessage;
  if (msg.length()>26) msg=msg.substring(0,26);
  tft.getTextBounds(msg.c_str(),0,0,&x1,&y1,&tw,&th);
  tft.setCursor(14+(SW-28-tw)/2,104); tft.print(msg);

  int bx=34,by=136,bw=SW-68,bh=8;
  tft.fillRect(bx,by,bw,bh,GRAY);
  float conf=lastConf.toFloat();
  int fill=(int)(conf*bw);
  if (fill>0) tft.fillRect(bx,by,fill,bh,col);
  tft.drawRect(bx,by,bw,bh,LGRAY);
}

static void drawInfoRows() {
  tft.fillRect(0,156,SW,98,BLACK);
  tft.drawFastHLine(0,156,SW,GRAY);

  tft.setTextSize(2);
  tft.setTextColor(LGRAY,BLACK); tft.setCursor(14,164); tft.print("NAME ");
  tft.setTextColor(WHITE,BLACK); tft.setCursor(100,164);
  String n=(lastName.length()>0)?lastName:"-";
  if (n.length()>18) n=n.substring(0,18);
  tft.print(n);

  tft.setTextColor(LGRAY,BLACK); tft.setCursor(14,196); tft.print("CONF ");
  tft.setTextColor(CYAN,BLACK);  tft.setCursor(100,196); tft.print(lastConf);

  tft.drawFastHLine(0,254,SW,GRAY);
}

static void drawHomeScreen() {
  tft.fillScreen(BLACK);
  drawHomeHeader();
  drawStatusCard();
  drawInfoRows();
  fillBtn(hBtn_ManualEnroll, TEAL,   CYAN,  "MANUAL ENROLL", WHITE, 2);
  fillBtn(hBtn_Clear,        0x2104, LGRAY, "CLEAR",         LGRAY, 2);
}

static void refreshHome() {
  drawStatusCard();
  drawInfoRows();
}

// ════════════════════════════════════════
// CONFIRM ENROLL SCREEN
// Shown automatically when Python sends UI|ENROLL_PROMPT|
// ════════════════════════════════════════
Btn cBtn_Enroll = { 30,  252, 190, 52 };
Btn cBtn_Ignore = { 260, 252, 190, 52 };

static void drawConfirmScreen() {
  tft.fillScreen(BLACK);

  // Header
  tft.fillRect(0,0,SW,46,DYELLOW);
  tft.drawFastHLine(0,46,SW,YELLOW);
  tft.setTextSize(2); tft.setTextColor(BLACK,DYELLOW);
  tft.setCursor(14,14); tft.print("!! UNKNOWN IRIS DETECTED !!");

  // Big question mark
  tft.setTextSize(8); tft.setTextColor(YELLOW,BLACK);
  tft.setCursor(388,60); tft.print("?");

  // Message lines
  tft.setTextSize(2); tft.setTextColor(WHITE,BLACK);
  tft.setCursor(14,60);  tft.print("A new iris was scanned.");
  tft.setCursor(14,86);  tft.print("This person is NOT enrolled.");
  tft.setCursor(14,120); tft.print("Do you want to enroll");
  tft.setCursor(14,146); tft.print("this person now?");

  tft.setTextSize(1); tft.setTextColor(LGRAY,BLACK);
  tft.setCursor(14,180); tft.print("The last captured frame will be used for enrollment.");
  tft.setCursor(14,194); tft.print("Make sure the eye is clearly positioned.");

  tft.drawFastHLine(14,216,SW-28,GRAY);

  fillBtn(cBtn_Enroll, DGREEN, GREEN, "YES  ENROLL", WHITE, 2);
  fillBtn(cBtn_Ignore, DRED,   RED,   "NO  IGNORE",  WHITE, 2);
}

// ════════════════════════════════════════
// KEYBOARD SCREEN
// ════════════════════════════════════════
#define KW 42
#define KH 36
#define KG 2

const char* ROW1 = "QWERTYUIOP";
const char* ROW2 = "ASDFGHJKL";
const char* ROW3 = "ZXCVBNM";

Btn eBtn_Back = { 14,  258, 120, 44 };
Btn eBtn_Del  = { 160, 258, 130, 44 };
Btn eBtn_Ok   = { 316, 258, 150, 44 };

static int kx(int row,int i){
  if(row==0) return 4  +i*(KW+KG);
  if(row==1) return 26 +i*(KW+KG);
  if(row==2) return 70 +i*(KW+KG);
  return 0;
}
static int ky(int row){ return 88+row*(KH+KG); }

static void drawOneKey(int x,int y,const char* txt,uint16_t bg){
  tft.fillRoundRect(x,y,KW,KH,3,bg);
  tft.drawRoundRect(x,y,KW,KH,3,LGRAY);
  tft.setTextSize(2); tft.setTextColor(WHITE,bg);
  int16_t x1,y1; uint16_t tw,th;
  tft.getTextBounds(txt,0,0,&x1,&y1,&tw,&th);
  tft.setCursor(x+(KW-tw)/2, y+(KH-th)/2);
  tft.print(txt);
}

static void drawKeyboardScreen(){
  tft.fillScreen(BLACK);

  // Header
  tft.fillRect(0,0,SW,40,0x2820);
  tft.drawFastHLine(0,40,SW,ORANGE);
  tft.setTextSize(2); tft.setTextColor(ORANGE,0x2820);
  tft.setCursor(14,12); tft.print("ENTER NAME TO ENROLL");

  // Input box
  tft.fillRoundRect(14,44,SW-28,38,4,0x0841);
  tft.drawRoundRect(14,44,SW-28,38,4,ORANGE);
  tft.setTextSize(2); tft.setTextColor(WHITE,0x0841);
  String shown=enrollName;
  if(shown.length()>22) shown=shown.substring(shown.length()-22);
  tft.setCursor(22,54); tft.print(shown+"_");

  for(int i=0;i<10;i++){char s[2]={ROW1[i],0}; drawOneKey(kx(0,i),ky(0),s,DBLUE);}
  for(int i=0;i<9; i++){char s[2]={ROW2[i],0}; drawOneKey(kx(1,i),ky(1),s,DBLUE);}
  for(int i=0;i<7; i++){char s[2]={ROW3[i],0}; drawOneKey(kx(2,i),ky(2),s,DBLUE);}

  int sy=ky(3);
  tft.fillRoundRect(100,sy,280,KH,3,GRAY); tft.drawRoundRect(100,sy,280,KH,3,LGRAY);
  tft.setTextSize(2); tft.setTextColor(WHITE,GRAY);
  tft.setCursor(220,sy+10); tft.print("SPC");

  fillBtn(eBtn_Back, 0x2104, LGRAY, "BACK",   WHITE, 2);
  fillBtn(eBtn_Del,  DRED,   RED,   "DEL",    WHITE, 2);
  fillBtn(eBtn_Ok,   DGREEN, GREEN, "ENROLL", WHITE, 2);
}

static void refreshInput(){
  tft.fillRoundRect(15,45,SW-30,36,3,0x0841);
  tft.setTextSize(2); tft.setTextColor(WHITE,0x0841);
  String shown=enrollName;
  if(shown.length()>22) shown=shown.substring(shown.length()-22);
  tft.setCursor(22,54); tft.print(shown+"_");
}

static void handleKeyboardTouch(int x,int y){
  if(hit(x,y,eBtn_Back)){
    screen=CONFIRM_ENROLL; drawConfirmScreen(); return;
  }
  if(hit(x,y,eBtn_Del)){
    if(enrollName.length()>0) enrollName.remove(enrollName.length()-1);
    refreshInput(); return;
  }
  if(hit(x,y,eBtn_Ok)){
    String n=enrollName; n.trim();
    if(n.length()==0) return;
    // feedback
    tft.fillRoundRect(14,44,SW-28,38,4,DGREEN);
    tft.drawRoundRect(14,44,SW-28,38,4,GREEN);
    tft.setTextSize(2); tft.setTextColor(WHITE,DGREEN);
    tft.setCursor(22,54); tft.print("Enrolling: "); tft.print(n);
    Serial.println("ENROLL:"+n);
    delay(400);
    screen=HOME; drawHomeScreen(); return;
  }

  char s[2]={0,0};
  for(int i=0;i<10;i++){
    int rx=kx(0,i),ry=ky(0);
    if(x>=rx&&x<rx+KW&&y>=ry&&y<ry+KH){
      if(enrollName.length()<24){s[0]=ROW1[i];enrollName+=s;}
      refreshInput();return;
    }
  }
  for(int i=0;i<9;i++){
    int rx=kx(1,i),ry=ky(1);
    if(x>=rx&&x<rx+KW&&y>=ry&&y<ry+KH){
      if(enrollName.length()<24){s[0]=ROW2[i];enrollName+=s;}
      refreshInput();return;
    }
  }
  for(int i=0;i<7;i++){
    int rx=kx(2,i),ry=ky(2);
    if(x>=rx&&x<rx+KW&&y>=ry&&y<ry+KH){
      if(enrollName.length()<24){s[0]=ROW3[i];enrollName+=s;}
      refreshInput();return;
    }
  }
  int sy=ky(3);
  if(x>=100&&x<380&&y>=sy&&y<sy+KH){
    if(enrollName.length()<24&&enrollName.length()>0
       &&enrollName[enrollName.length()-1]!=' ')
      enrollName+=' ';
    refreshInput();
  }
}

// ════════════════════════════════════════
// SERIAL
// ════════════════════════════════════════
static void handleUiLine(const String& line){
  int p1=line.indexOf('|');
  int p2=line.indexOf('|',p1+1);
  if(p2<0) return;
  String key=line.substring(p1+1,p2); key.trim();
  String val=line.substring(p2+1);    val.trim();

  if      (key=="STATUS")  lastStatus =val;
  else if (key=="NAME")    lastName   =(val=="None"||val=="null")?"":val;
  else if (key=="CONF")    lastConf   =val;
  else if (key=="MESSAGE") lastMessage=val;

  else if (key=="ENROLL_PROMPT"){
    // Auto-popup: unknown iris detected
    screen=CONFIRM_ENROLL;
    drawConfirmScreen();
    return;
  }
  else if (key=="ENROLL_OK"){
    lastStatus ="enrolled";
    lastMessage="Enrolled: "+val;
    lastName   =val;
    notifyMsg  ="ENROLLED: "+val; notifyCol=GREEN; notifyUntil=millis()+3000;
    screen=HOME; drawHomeScreen(); return;
  }
  else if (key=="ENROLL_FAIL"){
    notifyMsg  ="FAIL: "+val; notifyCol=RED; notifyUntil=millis()+3000;
    lastMessage="Enroll failed: "+val;
    screen=HOME; drawHomeScreen(); return;
  }

  if(screen==HOME) refreshHome();
}

static void handleServerLine(String line){
  line.trim();
  if(line.length()==0) return;
  if(line.startsWith("UI|")){ handleUiLine(line); return; }

  // legacy: status|name|conf
  int p1=line.indexOf('|');
  int p2=(p1>=0)?line.indexOf('|',p1+1):-1;
  if(p1<0||p2<0) return;
  lastStatus=line.substring(0,p1);    lastStatus.trim();
  lastName  =line.substring(p1+1,p2); lastName.trim();
  lastConf  =line.substring(p2+1);    lastConf.trim();
  if(lastName=="None"||lastName=="null") lastName="";
  lastConf=String(lastConf.toFloat(),3);
  if(screen==HOME) refreshHome();
}

// ════════════════════════════════════════
// NOTIFY BANNER
// ════════════════════════════════════════
static void tickNotify(){
  if(screen!=HOME||notifyMsg.length()==0) return;
  if(millis()<notifyUntil){
    uint16_t bg=(notifyCol==GREEN)?DGREEN:DRED;
    tft.fillRect(0,0,SW,46,bg);
    tft.drawFastHLine(0,46,SW,notifyCol);
    tft.setTextSize(2); tft.setTextColor(WHITE,bg);
    int16_t x1,y1; uint16_t tw,th;
    tft.getTextBounds(notifyMsg.c_str(),0,0,&x1,&y1,&tw,&th);
    tft.setCursor((SW-tw)/2,14); tft.print(notifyMsg);
  } else {
    notifyMsg=""; drawHomeHeader();
  }
}

// ════════════════════════════════════════
// TOUCH
// ════════════════════════════════════════
static bool readTouch(int& sx,int& sy){
  TSPoint p=ts.getPoint();
  pinMode(XM,OUTPUT); pinMode(YP,OUTPUT);
  if(p.z<MINPRESSURE||p.z>MAXPRESSURE) return false;
  sx=constrain(map(p.x,TS_MINX,TS_MAXX,0,SW),0,SW-1);
  sy=constrain(map(p.y,TS_MINY,TS_MAXY,0,SH),0,SH-1);
  return true;
}

// ════════════════════════════════════════
// SETUP / LOOP
// ════════════════════════════════════════
void setup(){
  // Must match SERIAL_BAUD in server.py (was 9600 → garbage if PC uses 115200).
  Serial.begin(115200);
  uint16_t ID=tft.readID();
  if(ID==0xD3D3) ID=0x9486;
  tft.begin(ID); tft.setRotation(1);
  drawHomeScreen();
  Serial.println("GETUI");
}

void loop(){
  while(Serial.available()){
    char c=(char)Serial.read();
    if(c=='\n'){ handleServerLine(rxLine); rxLine=""; }
    else if(c!='\r'){ rxLine+=c; if(rxLine.length()>240) rxLine=""; }
  }

  tickNotify();

  int tx,ty;
  if(readTouch(tx,ty)){
    delay(160);
    if(screen==HOME){
      if(hit(tx,ty,hBtn_ManualEnroll)){
        enrollName=""; screen=KEYBOARD; drawKeyboardScreen();
      } else if(hit(tx,ty,hBtn_Clear)){
        lastStatus="waiting"; lastName=""; lastConf="0.000";
        lastMessage="Waiting for scan..."; drawHomeScreen();
      }
    }
    else if(screen==CONFIRM_ENROLL){
      if(hit(tx,ty,cBtn_Enroll)){
        enrollName=""; screen=KEYBOARD; drawKeyboardScreen();
      } else if(hit(tx,ty,cBtn_Ignore)){
        Serial.println("IGNORE");
        screen=HOME; drawHomeScreen();
      }
    }
    else if(screen==KEYBOARD){
      handleKeyboardTouch(tx,ty);
    }
  }
}
