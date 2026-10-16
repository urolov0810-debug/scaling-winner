# CLPT + FSDT + Rjanitsin–Koltunov — Windows desktop ilova (loyiha)

**Muhim:** Bu ZIP ichida **tayyor Windows `.exe` yoki `Setup.exe` yo'q** — bu Windows kompyuterda ularga yig'iladigan **to'liq loyiha**. Bizning sinov muhitimiz Windows emasligi sabab `.exe`ni shu yerda ishlab chiqarmadik. Ilova o'rnatilib bo'lgach, hisoblash uchun **internet va Google Colab kerak emas**. Birinchi marta yig'ish jarayonida Python kutubxonalarini yuklash uchun internet kerak bo'ladi.

## 1. Windows 10/11 da birinchi ishga tushirish

1. ZIP arxivini papkaga chiqarib oling.
2. **Python 3.11 yoki 3.12 (64-bit)** o'rnating. O'rnatishda `Add Python to PATH`ni tanlang; `py` Windows Python launcher ham o'rnatilgan bo'lsin.
3. Bu loyiha papkasida terminal (CMD/PowerShell) oching:

```bat
py -3.12 -m pip install -r requirements.txt
py -3.12 main.py
```

Agar faqat Python 3.11 bo'lsa, ikki buyruqda ham `-3.12` o'rniga `-3.11` yozing.

**GUI yordamida**: Geometriya, Qatlamlar, Yuklanish, Hisob parametrlari bo'limlarida qiymatlarni kiriting; **Hisoblash** tugmasini bosing. Dastur o'zi alohida protsessda hisoblaydi va natijalar tabida jadval/grafiklarni ko'rsatadi. Demo qiymatlari faqat **tahrirlash uchun**; foydalanuvchi tugma bosmaguncha avtomatik hisoblanmaydi.

## 2. Python/Colab talab qilmaydigan EXE qilish

**Windows x64 kompyuterda** `BUILD_WINDOWS.bat` faylini ikki marta bosing. Birinchi yig'ishda kutubxonalar internetdan o'rnatiladi. Natija:

`dist\CLPT_FSDT_RK\CLPT_FSDT_RK.exe`

Butun `dist\CLPT_FSDT_RK` papkasini boshqa mos Windows 10/11 x64 kompyuterga ko'chirsangiz, **Python o'rnatmasdan, internetsiz** `.exe` ishlashi kerak. Faqat `.exe` faylning o'zini alohida ko'chirmang: yonidagi fayl va kutubxonalar ham kerak.

## 3. Windows uchun Setup.exe o'rnatgich qilish

1. Avval `BUILD_WINDOWS.bat` bajarilsin.
2. **Inno Setup 6** dasturini Windows build kompyuterga bir marta o'rnating.
3. `BUILD_INSTALLER.bat` ni ikki marta bosing.
4. Natija: `installer_output\CLPT_FSDT_RK_Setup_Windows_x64.exe`.

Bu o'rnatgich kerakli kutubxonalarni birga olib yuradi; boshqa mos Windows kompyuterlarga **internet talab qilmasdan** o'rnatiladi. O'rnatgich odatda joriy foydalanuvchi uchun administrator huquqisiz o'rnatiladi; Windows SmartScreen yoki korporativ siyosat ogohlantirishlari bo'lishi mumkin. Windows uchun EXE yig'ishni **Windowsning o'zida** bajaring: Linuxdagi PyInstaller Windows EXE yaratmaydi.

## 4. Hisoblash imkoniyatlari

- **CLPT / Ritz**: yupqa ortotrop qatlamli plastinkalar, RK xotira funksiyasi.
- **FSDT / MITC4 FEM**: 5 DOF, ko'ndalang kesish deformatsiyasi, RK ning Maxwell eksponentlar orqali yaqinlashuvi.
- **S/C/F tayanchlar**, **Winkler (kw)** va **Pasternak (kp)**.
- Har qatlamga `h, theta, E1, E2, G12, G13, G23, nu12, rho`; viskoelastik qatlamga `A, alpha, beta`.
- **Nuqtaviy, kontakt, bir tekis, garmonik va zarb yuklari**.
- **Chastotalar, salqilik, kuchlanish, mustahkamlik indeksi (FSDT), grafiklar**, solishtirish.

**Eslatma:** CLPT va FSDT alohida hisoblash usullari; ularning matritsalari qo'shilmaydi. SSSS bo'yicha ularning tayanch kinematik farazlari ham turlicha. Natijalarni konvergentsiya, material ma'lumotlari va etalon bilan tekshirish shart. Bu dastur ANSYS/Abaqusning barcha imkoniyatlarini almashtirmaydi.

## 5. Saqlash va natija papkalari

- **Fayl → JSON saqlash** orqali loyiha parametrlarini saqlang; **JSON ochish** orqali qayta yuklang.
- Har bir yangi hisoblash uchun alohida nomli `hisob_YYYYMMDD_HHMMSS_...` papkasi yaratiladi.
- Odatda `Documents\CLPT_FSDT_RK\hisob_...` ichida `kiritilgan_parametrlar.json`, `fsdt\` va/yoki `clpt\`, grafik va CSV fayllar mavjud bo'ladi.
- **Natijalar** tabida jurnal, asosiy chastotalar/grafik va papkani ochish tugmasi bor.
- **To'xtatish** tugmasi alohida hisoblash protsessini tugatadi. To'xtatilgan hisobning qisman fayllaridan muhandislik xulosasi qilmang.

## 6. Tuzilma

| Fayl | Vazifasi |
|---|---|
| `main.py` | GUI start va hisoblash child-process rejimi |
| `gui.py` | Windows uslubidagi Tkinter/ttk interfeys |
| `app_config.py` | Parametrlarni o'qish/tekshirish/saqlash |
| `engine.py` | Oldingi birlashtirilgan CLPT + FSDT + RK hisoblash kodi |
| `namuna_parametrlar.json` | Tahrirlash uchun tayyor namuna |
| `BUILD_WINDOWS.bat` | PyInstaller orqali EXE yaratish |
| `installer.iss`, `BUILD_INSTALLER.bat` | Inno Setup orqali o'rnatgich |
| `tests/` | Konfiguratsiya va kichik hisobni sinash |

## 7. Sinov buyruqlari

```bat
py -3.12 -m unittest discover -s tests -v
py -3.12 main.py --self-test
```

### Cheklovlar

* Materiallar qatlamlararo mukammal biriktirilgan deb qaraladi; geometriya va deformatsiya chiziqli.
* Katta siljish, delaminatsiya, plastik deformatsiya va murakkab 3D tahlil yo'q.
* FSDT nuqtaviy kuchga yaqin lokal mustahkamlik xulosasi ishonchli bo'lmasligi mumkin; patch yukini qo'llang.
* 64-bit Windows 10/11 uchun Python 3.11 yoki 3.12 tavsiya etiladi.

## 8. Kompyuterda Python bo'lmasa: GitHub Actions orqali Setup.exe

`README_GITHUB_UZ.md` dagi qo'llanmani oching. Loyihaga `.github/workflows/windows-setup.yml`
qo'shilgan: GitHub Actions Windows serverida test + PyInstaller + Inno Setup bosqichlarini
avtomatik bajaradi, natijadagi `Setup.exe`ni GitHub sahifasidan yuklab olish mumkin.
