#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hdual.h"

short hdual::counter = 0;
fuzzy** hdual::errors = NULL;

char getbit[8] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };

hdual::hdual(type real, type inf, ErrorInfo *Info)
   {
   this->ptr_size = 0;
   this->inf_size = 1.0;
   this->inf = NULL;
   this->inf_ptr = NULL;
   if (inf)
      {
      short i;
      this->ptr_size = this->counter++;

      //-----------------------------------------------
      fuzzy **temp = new fuzzy*[this->counter];
      for (i = 0; i < this->counter-1; i++)
	 temp[i] = this->errors[i];
      delete(this->errors);
      temp[this->counter-1] = new fuzzy(Info);
      this->errors = temp;
      //-----------------------------------------------

      this->inf_size = 1;
      this->inf = new type[1];
      this->inf[0] = inf;

      short inf_ptr_size = this->ptr_size >> 3;
      char  bit_number = this->ptr_size++ & 0x0007;
      this->inf_ptr = new char[inf_ptr_size + 1];

      for (i = 0; i < inf_ptr_size; i++)
	this->inf_ptr[i] = 0;
      this->inf_ptr[i] = (1 << bit_number);
      }
   this->real = real;
   }

hdual::~hdual(void)
   {
   if(this->inf_size)
      {
      delete(this->inf);
      delete(this->inf_ptr);
      }
   }

hdual::hdual(hdual& arg)
   {
   this->real = arg.real;
   this->inf_size = arg.inf_size;
   this->ptr_size = arg.ptr_size;
   this->inf = new type[this->inf_size];

   short inf_ptr_size = (this->ptr_size - 1) >> 3;
   this->inf_ptr = new char[inf_ptr_size + 1];

   short i;
   for (i = 0; i < this->inf_size; i++)
     this->inf[i] = arg.inf[i];
   for (i = 0; i <= inf_ptr_size; i++)
     this->inf_ptr[i] = arg.inf_ptr[i];
   }

void hdual::operator =(hdual& arg)
   {
   this->real = arg.real;
   if(this->inf_size)
      {
      delete(this->inf);
      delete(this->inf_ptr);
      }
   this->inf_size = arg.inf_size;
   this->ptr_size = arg.ptr_size;
   this->inf = new type[this->inf_size];

   short inf_ptr_size = (this->ptr_size - 1) >> 3;
   this->inf_ptr = new char[inf_ptr_size + 1];

   short i;
   for (i = 0; i < this->inf_size; i++)
     this->inf[i] = arg.inf[i];
   for (i = 0; i <= inf_ptr_size; i++)
     this->inf_ptr[i] = arg.inf_ptr[i];
   }

hdual operator +(hdual& arg, type constant)
   {
   hdual result(arg);
   result.real += constant;
   return result;
   }

hdual operator +(type constant, hdual& arg)
   {
   hdual result(arg);
   result.real += constant;
   return result;
   }

hdual operator +(hdual& f, hdual& g)
   {
   /* Объем массива для инфинитезимальных составляющих дуального
   результата операции сложения есть число единичных битов в маске
   f.inf_ptr | g.inf_ptr. */

   hdual result(f.real + g.real);

   short f_inf_ptr_size = f.ptr_size >> 3;
   short g_inf_ptr_size = g.ptr_size >> 3;
   char  f_bit_number = f.ptr_size & 0x0007;  /* Последние три бита. */
   char  g_bit_number = g.ptr_size & 0x0007;  /* Последние три бита. */
   if (f_bit_number)
      f_inf_ptr_size++;
   if (g_bit_number)
      g_inf_ptr_size++;

   char j;
   type buffer;
   short min_size, max_size;
   short i, sum = 0;
   hdual *arg_min;
   hdual *arg_max;
   if (f_inf_ptr_size < g_inf_ptr_size)
      {
      arg_min  = &f;
      arg_max  = &g;
      min_size = f_inf_ptr_size;
      max_size = g_inf_ptr_size;
      }
   else
      {
      arg_max  = &f;
      arg_min  = &g;
      max_size = f_inf_ptr_size;
      min_size = g_inf_ptr_size;
      }

   result.inf_ptr = new char[max_size];

   for (i = 0; i < min_size; i++)
      {
      result.inf_ptr[i] = f.inf_ptr[i] | g.inf_ptr[i];
      for (j = 0; j < 8; j++)
	 {
	 if (result.inf_ptr[i] & getbit[j])
	    sum = sum + 1;
	 }
      }
   for (i = min_size; i < max_size; i++)
      {
      result.inf_ptr[i] = arg_max->inf_ptr[i];
      for (j = 0; j < 8; j++)
	 {
	 if (result.inf_ptr[i] & getbit[j])
	    sum = sum + 1;
	 }
      }

   result.inf = new type[sum];
   result.inf_size = sum;
   result.ptr_size = max(f.ptr_size, g.ptr_size);

   short max_index = 0;
   short min_index = 0;
   short result_index = 0;
   for (i = 0; i < min_size; i++)
      {
      for (j = 0; j < 8; j++)
	 {
	 if (!(result.inf_ptr[i] & getbit[j]))
	    continue;
	 buffer = 0;
	 if (arg_max->inf_ptr[i] & getbit[j])
	    buffer += arg_max->inf[max_index++];
	 if (arg_min->inf_ptr[i] & getbit[j])
	    buffer += arg_min->inf[min_index++];
	 result.inf[result_index++] = buffer;
	 }
      }
   for (i = max_index; i < arg_max->inf_size; i++)
      result.inf[result_index++] = arg_max->inf[i];
   return result;
   }

hdual operator -(hdual& arg, type constant)
   {
   hdual result(arg);
   result.real -= constant;
   return result;
   }

hdual operator -(type constant, hdual& arg)
   {
   hdual result(arg);
   result = (-1) * result;
   result.real += constant;
   return result;
   }

hdual operator -(hdual& f, hdual& g)
   {
   hdual g_copy(g);
   short i;
   for (i = 0; i < g.inf_size; i++)
      g_copy.inf[i] *= - 1;
   g_copy.real *= - 1;
   return (f + g_copy);
   }

hdual operator *(hdual& f, hdual& g)
   {
   /* Объем массива для инфинитезимальных составляющих дуального
   резульата операции сложения есть число единичных битов в маске
   f.inf_ptr | g.inf_ptr. */

   hdual result(f.real * g.real);

   short f_inf_ptr_size = f.ptr_size >> 3;
   short g_inf_ptr_size = g.ptr_size >> 3;
   char  f_bit_number = f.ptr_size & 0x0007;  /* Последние три бита. */
   char  g_bit_number = g.ptr_size & 0x0007;  /* Последние три бита. */
   if (f_bit_number)
      f_inf_ptr_size++;
   if (g_bit_number)
      g_inf_ptr_size++;

   char j;
   type buffer;
   short min_size, max_size;
   short i, sum = 0;
   hdual *arg_min;
   hdual *arg_max;
   if (f_inf_ptr_size < g_inf_ptr_size)
      {
      arg_min  = &f;
      arg_max  = &g;
      min_size = f_inf_ptr_size;
      max_size = g_inf_ptr_size;
      }
   else
      {
      arg_max  = &f;
      arg_min  = &g;
      max_size = f_inf_ptr_size;
      min_size = g_inf_ptr_size;
      }

   result.inf_ptr = new char[max_size];

   for (i = 0; i < min_size; i++)
      {
      result.inf_ptr[i] = f.inf_ptr[i] | g.inf_ptr[i];
      for (j = 0; j < 8; j++)
	 {
	 if (result.inf_ptr[i] & getbit[j])
	    sum = sum + 1;
	 }
      }
   for (i = min_size; i < max_size; i++)
      {
      result.inf_ptr[i] = arg_max->inf_ptr[i];
      for (j = 0; j < 8; j++)
	 {
	 if (result.inf_ptr[i] & getbit[j])
	    sum = sum + 1;
	 }
      }

   result.inf = new type[sum];
   result.inf_size = sum;
   result.ptr_size = max(f.ptr_size, g.ptr_size);

   short max_index = 0;
   short min_index = 0;
   short result_index = 0;
   for (i = 0; i < min_size; i++)
      {
      for (j = 0; j < 8; j++)
	 {
	 if (!(result.inf_ptr[i] & getbit[j]))
	    continue;
	 buffer = 0;
	 if (arg_max->inf_ptr[i] & getbit[j])
	    buffer += arg_max->inf[max_index++] * arg_min->real;
	 if (arg_min->inf_ptr[i] & getbit[j])
	    buffer += arg_min->inf[min_index++] * arg_max->real;
	 result.inf[result_index++] = buffer;
	 }
      }
   for (i = max_index; i < arg_max->inf_size; i++)
      result.inf[result_index++] = arg_max->inf[i] * arg_min->real;
   return result;
   }

hdual operator *(type constant, hdual& arg)
   {
   short i;
   hdual result (arg);
   result.real *= constant;
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= constant;
   return result;
   }

hdual operator *(hdual& arg, type constant)
   {
   short i;
   hdual result (arg);
   result.real *= constant;
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= constant;
   return result;
   }

hdual operator /(hdual& f, hdual& g)
   {
   if (g.real == 0)
      printf("Divide by zero.");

   short i;
   hdual g_copy (g);
   g_copy.real = 1.0f / g_copy.real;
   for (i = 0; i < g_copy.inf_size; i++)
      g_copy.inf[i] *= -g_copy.real * g_copy.real;
   return (f * g_copy);
   }

hdual operator /(type constant, hdual& arg)
   {
   if (arg.real == 0)
      printf("Divide by zero.");

   short i;
   type buffer;
   hdual result (arg);
   buffer = 1.0f / result.real;
   result.real = buffer * constant;
   buffer *= result.real;
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= - buffer;
   return result;
   }

hdual operator /(hdual& arg, type constant)
   {
   if (constant == 0)
      printf("Divide by zero.");

   short i;
   hdual result (arg);
   result.real /= constant;
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] /= constant;
   return result;
   }

void hdual::operator +=(hdual& arg)
   { *this = (*this) + arg; }

void hdual::operator -=(hdual& arg)
   { *this = (*this) - arg; }

void hdual::operator *=(hdual& arg)
   { *this = (*this) * arg; }

void hdual::operator /=(hdual& arg)
   { *this = (*this) / arg; }

/*----------------------------------*/
/*---- Mathematical functions. -----*/
hdual  acos  (hdual& arg)
   {
   short i;
   hdual result(arg);
   type derivative = 1.0f / sqrt(1.0f - result.real * result.real);
   result.real = acos(result.real);
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= - derivative;
   return result;
   }

hdual  asin  (hdual& arg)
   {
   short i;
   hdual result(arg);
   type derivative = 1.0f / sqrt(1.0f - result.real * result.real);
   result.real = asin(result.real);
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  atan  (hdual& arg)
   {
   short i;
   hdual result(arg);
   type derivative = 1.0f / (1.0f + result.real * result.real);
   result.real = atan(result.real);
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  atan2 (hdual& f, hdual& g)
   { return atan(f / g); }

hdual  cos   (hdual& arg)
   {
   short i;
   hdual result(arg);
   type derivative = -sin(result.real);
   result.real = cos(result.real);
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  cosh  (hdual& arg)
   {
   short i;
   hdual result(arg);
   type derivative = sinh(result.real);
   result.real = cosh(result.real);
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  exp   (hdual& arg)
   {
   short i;
   hdual result(arg);
   type derivative = exp(result.real);
   result.real = derivative;
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  fabs  (hdual& arg)
   {
   short i;
   hdual result(arg);
   if (result.real < 0)
      {
      for (i = 0; i < result.inf_size; i++)
	 result.inf[i] *= -1;
      }
   #ifdef DOUBLE_DEFINED
   result.real = fabs(arg.real);
   #endif

   #ifdef LONGDOUBLE_DEFINED
   result.real = fabsl(arg.real);
   #endif

   return result;
   }

hdual  log   (hdual& arg)
   {
   short i;
   hdual result(arg);
   type derivative = 1.0 / result.real;
   result.real = log(result.real);
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  log10 (hdual& arg)
   {
   short i;
   hdual result(arg);
   type derivative = M_LN10 / result.real;
   result.real = log10(result.real);
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  pow   (hdual& arg, type x)
   {
   short i;
   hdual result(arg);
   type buffer = pow(result.real, x - 1.0f);
   type derivative = x * buffer;
   result.real = buffer * result.real;
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  pow   (type x, hdual& arg)
   {
   short i;
   hdual result(arg);
   result.real = pow(x, result.real);
   type derivative = result.real * log(x);
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  pow   (hdual& f, hdual& g)
   { return (exp(g * log(f))); }

hdual  sin   (hdual& arg)
   {
   short i;
   hdual result(arg);
   type derivative = cos(result.real);
   result.real = sin(result.real);
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  sinh  (hdual& arg)
   {
   short i;
   hdual result(arg);
   type derivative = cosh(result.real);
   result.real = sinh(result.real);
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  sqrt  (hdual& arg)
   {
   short i;
   hdual result(arg);
   result.real = sqrt(result.real);
   type derivative = 0.5f / result.real;
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  tan   (hdual& arg)
   {
   short i;
   hdual result(arg);
   result.real = tan(result.real);
   type derivative = 1.0f + result.real * result.real;
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

hdual  tanh  (hdual& arg)
   {
   short i;
   hdual result(arg);
   result.real = tanh(result.real);
   type derivative = 1.0f - result.real * result.real;
   for (i = 0; i < result.inf_size; i++)
      result.inf[i] *= derivative;
   return result;
   }

fuzzy hdual::GetErrorFuzzyVariable (void)
   {
   int i = 0;
   fuzzy Sum;
   for (i = 0; i < this->inf_size; i++)
      Sum += this->inf[i] * (*this->errors[i]);
   return (Sum);
   }

interval hdual::GetErrorInterval (type alpha)
   {
   fuzzy Error = this->GetErrorFuzzyVariable();
   interval Result = Error.getinterval(alpha);
   Result.a += this->real;
   Result.b += this->real;
   return (Result);
   }

char hdual::operator >= (type arg)
   { return (this->real >= arg); }

char hdual::operator >(type arg)
   { return (this->real > arg); }

char hdual::operator <=(type arg)
   { return (this->real <= arg); }

char hdual::operator <(type arg)
   { return (this->real < arg); }
