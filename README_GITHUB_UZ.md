# GitHub orqali Python o'rnatmasdan Setup.exe yaratish

**Kompyuteringizga Python yoki Inno Setup o'rnatishingiz shart emas.**
Ular GitHub Actions'ning vaqtinchalik **Windows** kompyuterida o'rnatiladi.
Hisoblash dasturining o'zi internetga ulanmaydi; faqat GitHub'dagi yig'ish jarayoni internet talab qiladi.

## 1. GitHub repositoriy yaratish

1. https://github.com ga kiring (hisob bo'lmasa ro'yxatdan o'ting).
2. `+` -> `New repository` bosing, masalan, `CLPT-FSDT-RK-Desktop` deb nomlang.
3. Maxfiy kod/material bo'lsa **Private** tanlang. Bo'sh repositoriy yarating (README tayyorlash shart emas).

## 2. Loyiha fayllarini yuklash

1. Shu ZIPni oching. Ichidagi `main.py`, `engine.py`, `gui.py`, `app_config.py`,
   `installer.iss`, `requirements*.txt`, `namuna_parametrlar.json`, `tests/` va qolgan
   fayllarni GitHub repozitoriyning **asosiy (root) joyiga** yuklang.
2. `Upload files` oynasi papkani qabul qilmasa, `tests/test_project.py` va
   `tests/create_smoke_config.py` fayllarini GitHub ichida `Add file -> Create new file`
   orqali tegishli yo'l bilan yarating.
3. Eng muhimi: `.github/workflows/windows-setup.yml` aynan shu **yo'lda** bo'lsin.
   GitHub yashirin `.github` papkasini yuklashga ruxsat bermasa,
   `Add file -> Create new file` orqali fayl nomiga
   `.github/workflows/windows-setup.yml` yozib, ZIP ichidagi matnini joylashtiring.
4. `Commit changes` tugmasini bosing. Fayllarni joylashganda **butun ZIP faylni**
   yuklamang: ZIP ichidagi alohida fayllar kerak.

**Fayl tuzilishi** (asosiy qismlar):

```
(repo root)/
  .github/workflows/windows-setup.yml
  main.py
  engine.py
  gui.py
  app_config.py
  installer.iss
  requirements.txt
  requirements-build.txt
  namuna_parametrlar.json
  tests/test_project.py
  tests/create_smoke_config.py
```

## 3. Bulutda yig'ish

1. GitHub'da `Actions` tabini oching (birinchi marta `Enable workflows` talab qilinishi mumkin).
2. Chapda `Windows Setup EXE` ni tanlang.
3. `Run workflow` -> `Run workflow` bosing (yoki `main` branchga yuklaganda avtomatik boshlanadi).
4. `Build and test Windows installer` qatori **yashil** bo'lguncha kuting.
5. Muvaffaqiyatli run sahifasining eng pastidagi **Artifacts** bo'limidan
   `CLPT_FSDT_RK_Setup_Windows_x64` ni yuklab oling.
6. GitHub ZIP artifaktini chiqarib, ichidan
   `CLPT_FSDT_RK_Setup_Windows_x64.exe` faylini oling.
7. Uni Windows 10/11 (64-bit) kompyuterida ochib o'rnating.

**Natija:** o'rnatilgan dastur Python/Colab va internetsiz hisoblaydi.
Installerning o'zi ham internetga muhtoj emas.

## Muammo chiqsa

* **Actions chiqmayapti:** `.github/workflows/windows-setup.yml` to'g'ri joylashganini tekshiring.
* **Qizil X:** Actions -> muvaffaqiyatsiz run -> qizil bosqichni oching, xato matnini nusxa oling.
* **tests not found:** `tests/` papkasini ham yuklang.
* **exe SmartScreen ogohlantirish:** bu kod-imzo sertifikati yo'q o'z-o'zidan yig'ilgan ilovalarda uchrashi mumkin;
  faqat o'zingiz ishongan faylni oching.
* **Repo maxfiyligi:** shaxsiy kod uchun Private repo tanlash mumkin. GitHub Actions hisob
  rejasiga qarab vaqt limiti va kvota ishlatadi.
* **Tekshiruv cheklovi:** loyiha sintaksisi va Python testlari tekshiriladi, ammo bu yerda
  real Windows Setup.exe yig'ilmadi. GitHub Windows runnerdagi ishga tushirish testi o'tsa,
  Windows buildning ishlashi haqida ishonch hosil qilish mumkin.
